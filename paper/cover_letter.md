<!-- Cover letter for the Sadhana submission (paste into Editorial Manager).
     Fill in the [brackets]; delete this comment. -->

[Date]

The Editor
Sādhanā – Academy Proceedings in Engineering Sciences
Indian Academy of Sciences, Bengaluru

Dear Editor,

I submit the manuscript **"Compact neural lemmatization for Tamil on edge devices"**
for consideration as a regular paper in Sādhanā (computer science).

Lemmatization, reducing an inflected word to its dictionary form, is a basic
step for search, spell checking and text analysis. Tamil, spoken by about
69 million people in India (Census 2011), is agglutinative, which makes the task
both necessary and hard, yet existing Tamil lemmatizers are either rule-based
analysers with limited coverage or neural models built for servers. The
manuscript asks how small and fast a Tamil lemmatizer can be on low-power
hardware, a question of direct relevance to Indian-language technology on
inexpensive devices.

The main contributions are:

1. a word–lemma dataset from the Universal Dependencies Tamil treebanks, split
   by lemma so that most test words have lemmas never seen in training, plus
   2,583 human-reviewed pairs for frequent Tamil Wikipedia words;
2. a controlled comparison of compact character models (0.5M–5M parameters,
   three seeds each), showing that with this amount of data accuracy does not
   grow with model size while knowledge distillation adds 6–8 points to an
   edit-script classifier;
3. a benchmark on a Raspberry Pi 5 across three inference runtimes and int8
   quantization: the distilled 0.5M-parameter model needs 538 KB and 0.29 ms per
   word, and with a 50,000-word lookup table it processes about 21,000 tokens per
   second at 82.5% token accuracy.

All code, data splits, annotations and trained models are publicly available
(https://github.com/khaleeljageer/edge-tam-lemma), and every number in the paper
is produced by scripts in that repository.

I confirm that the manuscript is original, has not been published, and is not
under consideration for publication elsewhere. [It has not been posted as a
preprint. / It has been posted as a preprint at arXiv:XXXX.XXXXX.] AI-assisted
tools were used as disclosed in the Declarations section of the manuscript
(Claude Code, Anthropic, with the model Claude Opus 5.5, for programming
assistance, running the experiment scripts and drafting the text); the author
checked every experiment, number and claim. There are no competing interests
and no funding to declare.

[Optional: suggested reviewers, with name, affiliation and e-mail, who have no
conflict of interest with the author.]

Yours sincerely,

SyedKhaleel Jageer
Independent Researcher, India
jskcse4@gmail.com
[ORCID: 0000-0000-0000-0000]
