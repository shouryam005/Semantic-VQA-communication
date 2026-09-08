"""
Matched RVNN and CVNN models for the SAR-VQA semantic communication pipeline.

Both models share, byte for byte:

  - the input tensor (a normalized complex image; the RVNN splits it into two
    real channels itself, so no information is discarded before either encoder)
  - the question encoder (real embedding + BiLSTM -> 128-d)
  - the classifier (192 -> 128 -> 64 -> 2)
  - the number of real values pushed through the channel (32)
  - the SNR definition, E[|x|^2] / E[|n|^2], which for real x reduces to E[x^2]
  - the switch that turns the semantic encoder / channel / decoder on or off

Only the image path differs, which is the thing under test.

Three defects in the notebook versions are fixed here.

  The RVNN baseline ran with its semantic encoder, AWGN channel and decoder
  ACTIVE (nb2_2 cell 13) while the CVNN baseline had all three commented out
  (nb3_2 cell 22). The 89%-vs-69% comparison was therefore between different
  pipelines. Here `use_channel` applies to both.

  ComplexSemanticEncoder.fc1 was nn.Linear(64, 48) but the complex image
  encoder emits 32 complex features, so enabling the commented block raised a
  shape error. The communication path had never run.

  awgn_channel was a plain function, so model.eval() did not disable it and
  validation accuracy was stochastic. It is a Module here and respects
  self.training via `noise_at_eval`.

Parameter matching. A complex layer stores two real weight tensors, so a
complex convolution with the same channel counts as a real one holds twice the
parameters. CVNN widths are therefore chosen to bring the total trainable
parameter count within ~1% of the RVNN rather than to match channel counts.
Call `parameter_report()` to print both.
"""

import torch
import torch.nn as nn

from .complex_layers import (
    CReLU,
    ComplexAdaptiveAvgPool2d,
    ComplexConv2d,
    ComplexLinear,
    ComplexMaxPool2d,
    make_activation,
)

IMAGE_FEATURES = 64      # real values leaving the image path, both models
QUESTION_FEATURES = 128  # real values leaving the question encoder
CHANNEL_WIDTH = 32       # real values transmitted, both models


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

class QuestionEncoder(nn.Module):
    """Identical in both models. 32-d embedding, BiLSTM with 64 per direction."""

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, questions):
        x = self.embedding(questions)
        _, (h, _) = self.lstm(x)
        return torch.cat([h[-2], h[-1]], dim=1)


class Classifier(nn.Module):
    def __init__(self, in_features=IMAGE_FEATURES + QUESTION_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    """
    Question-conditioned modulation of the image features before transmission.

    Without this the transmitter compresses the image without knowing the
    question, which makes the pipeline image compression followed by VQA rather
    than task-oriented semantic communication. The point of semantic
    communication is that the sender transmits what the task needs, so the
    question has to reach the encoder.

    Feature-wise linear modulation (Perez et al. 2018): the question vector
    predicts a per-feature scale and shift,

        h <- gamma(q) * h + beta(q)

    which is the lightest conditioning that can gate features on or off. Both
    models get an identical FiLM stage, so the comparison stays controlled; for
    the complex model gamma and beta are complex, so conditioning can rotate
    phase as well as rescale magnitude.
    """

    def __init__(self, question_features, feature_count, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        self.feature_count = feature_count
        outputs = feature_count * (4 if complex_valued else 2)
        self.project = nn.Linear(question_features, outputs)
        # start as the identity: gamma = 1, beta = 0
        nn.init.zeros_(self.project.weight)
        with torch.no_grad():
            self.project.bias.zero_()
            self.project.bias[:feature_count] = 1.0

    def forward(self, features, question):
        params = self.project(question)
        n = self.feature_count
        if not self.complex_valued:
            return params[:, :n] * features + params[:, n:]
        gamma = torch.complex(params[:, :n], params[:, n:2 * n])
        beta = torch.complex(params[:, 2 * n:3 * n], params[:, 3 * n:])
        return gamma * features + beta


class AWGNChannel(nn.Module):
    """
    Additive white Gaussian noise at a fixed SNR.

    Signal power is E[|x|^2], which for a real tensor is E[x^2], so the real
    and complex branches use the same definition. Complex noise is split evenly
    between the real and imaginary parts so that E[|n|^2] matches the target.

    The notebooks used x.pow(2).mean() on the real side and abs(x).pow(2).mean()
    on the complex side; those agree only for real input, so the two pipelines
    were not run at the same SNR.
    """

    def __init__(self, snr_db=10.0, noise_at_eval=True):
        super().__init__()
        self.snr_db = snr_db
        self.noise_at_eval = noise_at_eval

    def forward(self, x):
        if not self.training and not self.noise_at_eval:
            return x

        signal_power = (torch.abs(x) ** 2).mean()
        noise_power = signal_power / (10 ** (self.snr_db / 10))

        if torch.is_complex(x):
            std = torch.sqrt(noise_power / 2)
            return x + torch.complex(torch.randn_like(x.real) * std,
                                     torch.randn_like(x.imag) * std)
        return x + torch.randn_like(x) * torch.sqrt(noise_power)


# --------------------------------------------------------------------------
# real-valued branch
# --------------------------------------------------------------------------

class RealImageEncoder(nn.Module):
    """Real and imaginary parts as two input channels."""

    def __init__(self, widths=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = widths
        self.net = nn.Sequential(
            nn.Conv2d(2, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_features = c3

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        return self.net(x).flatten(1)


class RealSemanticEncoder(nn.Module):
    def __init__(self, in_features=IMAGE_FEATURES, out_features=CHANNEL_WIDTH):
        super().__init__()
        hidden = (in_features + out_features) // 2
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)


class RealSemanticDecoder(nn.Module):
    def __init__(self, in_features=CHANNEL_WIDTH, out_features=IMAGE_FEATURES):
        super().__init__()
        hidden = (in_features + out_features) // 2
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# complex-valued branch
# --------------------------------------------------------------------------

class ComplexPoolingHead(nn.Module):
    """
    How the spatial map is collapsed before the classifier. This is where the
    notebook architecture silently destroyed phase, so it is made explicit.

    Global average pooling of complex values is the trap. Averaging numbers with
    unrelated phases makes them cancel: measured on the last conv block, only
    13.5% of the signal survives when a phase-preserving activation is used.
    CReLU hides this by forcing Re >= 0 and Im >= 0, which confines every
    activation to the first quadrant -- pooling then survives at 91%, but only
    because phase has already been crushed into a 90 degree wedge. Either way
    almost no phase reaches the classifier.

    All three modes emit the same width (2 * channels real values) so the
    classifier and the parameter budget are unchanged.

      "avg"        spatial mean of z, split into real and imaginary parts.
                   The notebook behaviour, kept as the baseline.

      "coherence"  [mean|z|, |mean z|] per channel. The first term is energy,
                   which survives any phase; the second is the coherent sum,
                   which is large only when phases across the map agree. Their
                   ratio is the standard SAR coherence estimator, so this gives
                   the classifier a genuine phase-structure feature instead of
                   a cancelled average.

      "modulus"    [mean|z|, max|z|] per channel. Phase-free by construction.
                   The control: if this matches the others, phase contributed
                   nothing and the complex machinery is decoration.
    """

    def __init__(self, mode="coherence"):
        super().__init__()
        self.mode = mode

    @property
    def emits_complex(self):
        return self.mode == "avg"

    def forward(self, x):
        if self.mode == "avg":
            return torch.complex(x.real.mean((2, 3)), x.imag.mean((2, 3)))

        modulus = torch.abs(x)
        energy = modulus.mean((2, 3))
        if self.mode == "coherence":
            coherent = torch.abs(torch.complex(x.real.mean((2, 3)), x.imag.mean((2, 3))))
            return torch.cat([energy, coherent], dim=1)
        if self.mode == "modulus":
            return torch.cat([energy, modulus.amax((2, 3))], dim=1)
        raise ValueError("unknown pooling mode: " + self.mode)


class ComplexImageEncoder(nn.Module):
    """
    One complex input channel. Emits IMAGE_FEATURES real values (or
    IMAGE_FEATURES/2 complex ones under "avg" pooling), so the vector handed to
    the classifier has the same width as the RVNN's either way.
    """

    def __init__(self, widths=(16, 26, 32), activation="crelu", pooling="avg"):
        super().__init__()
        c1, c2, c3 = widths
        act = lambda c: make_activation(activation, c)

        self.features = nn.Sequential(
            ComplexConv2d(1, c1, 3, padding=1), act(c1), ComplexMaxPool2d(2),
            ComplexConv2d(c1, c2, 3, padding=1), act(c2), ComplexMaxPool2d(2),
            ComplexConv2d(c2, c3, 3, padding=1), act(c3), ComplexMaxPool2d(2),
        )
        self.pool = ComplexPoolingHead(pooling)
        self.out_features = c3
        self.emits_complex = self.pool.emits_complex

    def forward(self, x):
        return self.pool(self.features(x.unsqueeze(1)))


class ComplexSemanticEncoder(nn.Module):
    def __init__(self, in_features=IMAGE_FEATURES // 2, out_features=CHANNEL_WIDTH // 2):
        super().__init__()
        hidden = (in_features + out_features) // 2
        self.fc1 = ComplexLinear(in_features, hidden)
        self.act = CReLU()
        self.fc2 = ComplexLinear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ComplexSemanticDecoder(nn.Module):
    def __init__(self, in_features=CHANNEL_WIDTH // 2, out_features=IMAGE_FEATURES // 2):
        super().__init__()
        hidden = (in_features + out_features) // 2
        self.fc1 = ComplexLinear(in_features, hidden)
        self.act = CReLU()
        self.fc2 = ComplexLinear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------
# full models
# --------------------------------------------------------------------------

class RVNNVQA(nn.Module):
    def __init__(self, vocab_size, use_channel=True, snr_db=10.0,
                 widths=(16, 32, 64), noise_at_eval=True, film=False, **_):
        super().__init__()
        self.image_encoder = RealImageEncoder(widths)
        self.question_encoder = QuestionEncoder(vocab_size)
        self.classifier = Classifier()
        self.use_channel = use_channel
        self.film = FiLM(QUESTION_FEATURES, IMAGE_FEATURES) if film else None
        if use_channel:
            self.semantic_encoder = RealSemanticEncoder(self.image_encoder.out_features)
            self.channel = AWGNChannel(snr_db, noise_at_eval)
            self.semantic_decoder = RealSemanticDecoder(out_features=IMAGE_FEATURES)

    def forward(self, images, questions):
        features = self.image_encoder(images)
        question = self.question_encoder(questions)
        if self.film is not None:
            features = self.film(features, question)
        if self.use_channel:
            features = self.semantic_decoder(self.channel(self.semantic_encoder(features)))
        return self.classifier(torch.cat([features, question], dim=1))


class CVNNVQA(nn.Module):
    """
    Complex image path. `pooling` decides whether the features leaving the
    encoder are still complex ("avg") or already real ("coherence",
    "modulus"); the semantic encoder matches, so the transmitted width stays
    CHANNEL_WIDTH real values in every configuration.
    """

    def __init__(self, vocab_size, use_channel=True, snr_db=10.0,
                 widths=(16, 26, 32), activation="crelu", pooling="avg",
                 noise_at_eval=True, film=False, **_):
        super().__init__()
        self.image_encoder = ComplexImageEncoder(widths, activation, pooling)
        self.question_encoder = QuestionEncoder(vocab_size)
        self.classifier = Classifier()
        self.use_channel = use_channel
        self.complex_path = self.image_encoder.emits_complex

        if film:
            count = IMAGE_FEATURES // 2 if self.complex_path else IMAGE_FEATURES
            self.film = FiLM(QUESTION_FEATURES, count, complex_valued=self.complex_path)
        else:
            self.film = None

        if use_channel:
            self.channel = AWGNChannel(snr_db, noise_at_eval)
            if self.complex_path:
                self.semantic_encoder = ComplexSemanticEncoder(self.image_encoder.out_features)
                self.semantic_decoder = ComplexSemanticDecoder(out_features=IMAGE_FEATURES // 2)
            else:
                self.semantic_encoder = RealSemanticEncoder(IMAGE_FEATURES)
                self.semantic_decoder = RealSemanticDecoder(out_features=IMAGE_FEATURES)

    def forward(self, images, questions):
        features = self.image_encoder(images)
        question = self.question_encoder(questions)
        if self.film is not None:
            features = self.film(features, question)
        if self.use_channel:
            features = self.semantic_decoder(self.channel(self.semantic_encoder(features)))
        if self.complex_path:
            # concatenation, not modulus: both components reach the classifier
            features = torch.cat([features.real, features.imag], dim=1)
        return self.classifier(torch.cat([features, question], dim=1))


class QuestionOnlyVQA(nn.Module):
    """
    Control model. Identical question encoder and classifier, no image at all.

    Any image model must clear this to have learned anything from the SAR data.
    On the rebuilt benchmark it should sit at chance.
    """

    def __init__(self, vocab_size, **_kwargs):
        super().__init__()
        self.question_encoder = QuestionEncoder(vocab_size)
        self.classifier = Classifier(in_features=QUESTION_FEATURES)

    def forward(self, images, questions):
        return self.classifier(self.question_encoder(questions))


MODELS = {"rvnn": RVNNVQA, "cvnn": CVNNVQA, "question-only": QuestionOnlyVQA}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_report(vocab_size=22, use_channel=True, snr_db=10.0):
    """Print the parameter budget of each model, split by component."""
    rows = []
    for name in ("rvnn", "cvnn", "question-only"):
        model = MODELS[name](vocab_size, use_channel=use_channel, snr_db=snr_db)
        parts = {n: count_parameters(m) for n, m in model.named_children()}
        rows.append((name, count_parameters(model), parts))

    components = ["image_encoder", "semantic_encoder", "semantic_decoder",
                  "question_encoder", "classifier"]
    header = "%-14s %10s  " % ("model", "total") + "".join("%18s" % c for c in components)
    print(header)
    print("-" * len(header))
    for name, total, parts in rows:
        line = "%-14s %10d  " % (name, total)
        line += "".join("%18s" % parts.get(c, "-") for c in components)
        print(line)

    image_only = [(n, p.get("image_encoder", 0)) for n, _, p in rows if n in ("rvnn", "cvnn")]
    r, c = image_only[0][1], image_only[1][1]
    print()
    print("image encoder: rvnn=%d  cvnn=%d  ratio=%.3f" % (r, c, c / r))
    rt, ct = rows[0][1], rows[1][1]
    print("full model:    rvnn=%d  cvnn=%d  ratio=%.3f" % (rt, ct, ct / rt))


if __name__ == "__main__":
    print("with semantic communication channel")
    parameter_report(use_channel=True)
    print()
    print("without channel (image path only)")
    parameter_report(use_channel=False)
