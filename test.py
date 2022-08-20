"""WMT17: Translate dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.translate import wmt

_URL = "http://www.statmt.org/wmt17/translation-task.html"
_CITATION = """
@InProceedings{bojar-EtAl:2017:WMT1,
  author    = {Bojar, Ond\v{r}ej  and  Chatterjee, Rajen  and  Federmann, Christian  and  Graham, Yvette  and  Haddow, Barry  and  Huang, Shujian  and  Huck, Matthias  and  Koehn, Philipp  and  Liu, Qun  and  Logacheva, Varvara  and  Monz, Christof  and  Negri, Matteo  and  Post, Matt  and  Rubino, Raphael  and  Specia, Lucia  and  Turchi, Marco},
  title     = {Findings of the 2017 Conference on Machine Translation (WMT17)},
  booktitle = {Proceedings of the Second Conference on Machine Translation, Volume 2: Shared Task Papers},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {169--214},
  url       = {http://www.aclweb.org/anthology/W17-4717}
}
"""

_LANGUAGE_PAIRS = [
    (lang, "en") for lang in ["cs", "de", "fi", "lv", "ru", "tr", "zh"]
]


class Wmt17Translate(wmt.WmtTranslate):
  """WMT 17 translation datasets for all {xx, "en"} language pairs."""

  BUILDER_CONFIGS = [
      wmt.WmtConfig(  # pylint:disable=g-complex-comprehension
          description="WMT 2017 %s-%s translation task dataset." % (l1, l2),
          url=_URL,
          citation=_CITATION,
          language_pair=(l1, l2),
          version=tfds.core.Version("1.0.0"),
      ) for l1, l2 in _LANGUAGE_PAIRS
  ]

  @property
  def _subsets(self):
    return {
        tfds.Split.TRAIN: [
            "europarl_v7", "europarl_v8_16", "commoncrawl",
            "newscommentary_v12", "czeng_16", "yandexcorpus",
            "wikiheadlines_fi", "wikiheadlines_ru", "setimes_2", "uncorpus_v1",
            "rapid_2016", "leta_v1", "dcep_v1", "onlinebooks_v1"
        ] + wmt.CWMT_SUBSET_NAMES,
        tfds.Split.VALIDATION: ["newsdev2017", "newstest2016", "newstestB2016"],
        tfds.Split.TEST: ["newstest2017", "newstestB2017"]
    }
