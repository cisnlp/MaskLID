# MaskLID

This repository maintains the code for 

**MaskLID: Code-Switching Language Identification through Iterative Masking** (ACL 2024): https://arxiv.org/abs/2406.06263

MaskLID is a simple, yet effective, code-switching (CS) language identification (LID) method. MaskLID does not require any training and is designed to complement current high-performance sentence-level LIDs. Sentence-level LIDs are classifiers trained on monolingual texts to provide single labels, typically using a softmax layer to turn scores into probabilities. However, in cases where a sentence is composed in both L1 and L2 languages, the LID classifier often only returns the dominant label L1. To address this limitation, MaskLID employs a strategy to **mask** text features associated with L1, allowing the LID to classify the text as L2 in the next round. This method uses the LID itself to identify the features that require masking and does not rely on any external resource.


## Demo


We host a demo of MaskLID with GlotLID on Hugging Face, in the **Code-Switching (MaskLID)** tab of the GlotLID Space: https://huggingface.co/spaces/cis-lmu/glotlid-space. You can choose GlotLID v3 or v4, control the parameters and see the words of each language.


## Code Python 

```python
# !pip install fasttext-turbo numpy threadpoolctl
# !pip install -U "GlotScript>=2.1"   # for scripts=True (used by the presets); until it is on PyPI:
# !pip install "GlotScript @ git+https://github.com/cisnlp/GlotScript"

# !wget https://raw.githubusercontent.com/cisnlp/MaskLID/main/masklid.py
# !wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v4.bin

```

[fastText-turbo](https://github.com/CommonGlot/fastText-turbo) is a drop-in replacement for `fasttext` (same `import fasttext`, same models) with faster prediction, and it works with NumPy 1 and 2. The original `fasttext` package works too; uninstall it before installing fasttext-turbo, as both provide the `fasttext` module. `threadpoolctl` is optional: with it, MaskLID runs NumPy's BLAS on one thread inside its calls, which is much faster for its small matrix products on machines with many cores.

```python
# masklid is the name of masklid.py
from masklid import MaskLID

# path to glotlid model
model_path = 'model_v3.bin'

# GlotLID has more than 2000 labels, here we limit the GlotLID to the 200 languages available in flores
flores_glotlid = ['__label__eng_Latn', '__label__arb_Arab', '__label__rus_Cyrl', '__label__por_Latn', '__label__pol_Latn', '__label__ekk_Latn', '__label__ell_Grek', '__label__slk_Latn', '__label__slv_Latn', '__label__nld_Latn', '__label__lvs_Latn', '__label__hun_Latn', '__label__dan_Latn', '__label__swe_Latn', '__label__lit_Latn', '__label__fin_Latn', '__label__mlt_Latn', '__label__cmn_Hani', '__label__nob_Latn', '__label__kor_Hang', '__label__ind_Latn', '__label__uzn_Latn', '__label__fil_Latn', '__label__ukr_Cyrl', '__label__hin_Deva', '__label__hin_Latn', '__label__afr_Latn', '__label__mar_Deva', '__label__ceb_Latn', '__label__ilo_Latn', '__label__zul_Latn', '__label__heb_Hebr', '__label__xho_Latn', '__label__vie_Latn', '__label__jpn_Jpan', '__label__guj_Gujr', '__label__hrv_Latn', '__label__tur_Latn', '__label__nya_Latn', '__label__tsn_Latn', '__label__sna_Latn', '__label__tso_Latn', '__label__tha_Thai', '__label__spa_Latn', '__label__deu_Latn', '__label__eus_Latn', '__label__bul_Cyrl', '__label__amh_Ethi', '__label__fra_Latn', '__label__ewe_Latn', '__label__mkd_Cyrl', '__label__nso_Latn', '__label__tam_Taml', '__label__lin_Latn', '__label__twi_Latn', '__label__yor_Latn', '__label__als_Latn', '__label__ibo_Latn', '__label__ben_Beng', '__label__ita_Latn', '__label__tpi_Latn', '__label__azj_Latn', '__label__run_Latn', '__label__mya_Mymr', '__label__kin_Latn', '__label__ron_Latn', '__label__ces_Latn', '__label__kat_Geor', '__label__urd_Arab', '__label__zsm_Latn', '__label__pap_Latn', '__label__bem_Latn', '__label__mal_Mlym', '__label__kir_Cyrl', '__label__hye_Armn', '__label__smo_Latn', '__label__sin_Sinh', '__label__fij_Latn', '__label__kan_Knda', '__label__pan_Guru', '__label__hau_Latn', '__label__epo_Latn', '__label__gaz_Latn', '__label__tir_Ethi', '__label__bos_Latn', '__label__srp_Cyrl', '__label__hat_Latn', '__label__pag_Latn', '__label__lua_Latn', '__label__war_Latn', '__label__tel_Telu', '__label__tat_Cyrl', '__label__sag_Latn', '__label__lug_Latn', '__label__tum_Latn', '__label__swh_Latn', '__label__umb_Latn', '__label__som_Latn', '__label__gle_Latn', '__label__kng_Latn', '__label__mos_Latn', '__label__lus_Latn', '__label__khk_Cyrl', '__label__asm_Beng', '__label__tuk_Latn', '__label__quy_Latn', '__label__ayr_Latn', '__label__luo_Latn', '__label__tgk_Cyrl', '__label__cat_Latn', '__label__ssw_Latn', '__label__nno_Latn', '__label__cym_Latn', '__label__kik_Latn', '__label__kmb_Latn', '__label__ory_Orya', '__label__bel_Cyrl', '__label__bho_Deva', '__label__apc_Arab', '__label__bak_Cyrl', '__label__jav_Latn', '__label__yue_Hani', '__label__pbt_Arab', '__label__khm_Khmr', '__label__npi_Deva', '__label__npi_Latn', '__label__gug_Latn', '__label__uig_Arab', '__label__fur_Latn', '__label__kbp_Latn', '__label__hne_Deva', '__label__kam_Latn', '__label__gla_Latn', '__label__kab_Latn', '__label__arz_Arab', '__label__kaz_Cyrl', '__label__mri_Latn', '__label__lim_Latn', '__label__srd_Latn', '__label__sun_Latn', '__label__plt_Latn', '__label__mni_Beng', '__label__isl_Latn', '__label__vec_Latn', '__label__glg_Latn', '__label__scn_Latn', '__label__fao_Latn', '__label__san_Deva', '__label__ltz_Latn', '__label__cjk_Latn', '__label__ast_Latn', '__label__lmo_Latn', '__label__szl_Latn', '__label__oci_Latn', '__label__fon_Latn', '__label__min_Latn', '__label__wol_Latn', '__label__lij_Latn', '__label__ajp_Arab', '__label__snd_Arab', '__label__dik_Latn', '__label__ary_Arab', '__label__lao_Laoo', '__label__ars_Arab', '__label__bjn_Latn', '__label__shn_Mymr', '__label__crh_Latn', '__label__aeb_Arab', '__label__ace_Latn', '__label__ckb_Arab', '__label__dyu_Latn', '__label__ltg_Latn', '__label__kmr_Latn', '__label__ban_Latn', '__label__mai_Deva', '__label__fuv_Latn', '__label__kac_Latn', '__label__taq_Latn', '__label__bam_Latn', '__label__sat_Olck', '__label__tzm_Tfng', '__label__bug_Latn', '__label__dzo_Tibt', '__label__kas_Deva', '__label__fas_Arab', '__label__nus_Latn', '__label__knc_Latn', '__label__mag_Deva', '__label__taq_Tfng', '__label__kas_Arab', '__label__knc_Arab', '__label__bjn_Arab', '__label__ace_Arab', '__label__kea_Latn', '__label__awa_Deva', '__label__acm_Arab', '__label__bod_Tibt', '__label__sot_Latn', '__label__ydd_Hebr', '__label__azb_Arab']

# if you want GlotLID to work for all languages, set 'languages' to -1.
masklid_model = MaskLID(model_path, languages = flores_glotlid)
```

```python
# now that you have created the masklid_model object, you can run the `predict_codeswitch` method as many times as you want.
text = "bir kahve dükkanında geçen film tadında güzel bir şarkıya ayrılsın gece falling in love at a coffee shop"
ans = masklid_model.predict_codeswitch(text, beta = 20 , alpha = 3, max_lambda = 3, min_length = 10, min_prob = 0.90, max_retry=3, alpha_step_increase = 3, beta_step_increase = 5)
ans
>> {'__label__tur_Latn': 'bir kahve dükkanında geçen tadında güzel bir şarkıya ayrılsın gece',
 '__label__eng_Latn': 'film falling in love at coffee shop'}
```


### Recommended settings: assign every word (`assign="switch"`)

With the paper's settings, MaskLID leaves words that no round took unassigned (about a quarter of the words on the
benchmarks below), and with `min_prob=0.9` it often misses short inserts of a second language. With
`assign="switch"`, MaskLID only finds the languages; every word is then assigned to one of them by a Viterbi pass
that adds up the words' log-probabilities and pays `switch_penalty` for each change of language between neighbouring
words, so short ambiguous words follow their neighbours. A language is kept if its words have at least `min_chars`
characters. Two presets are tuned for GlotLID v4 with all its labels:

```python
from masklid import MaskLID, MASKLID_BALANCED, MASKLID_CAUTIOUS
masklid_model = MaskLID('model_v4.bin')          # all labels
ans = masklid_model.predict_codeswitch(text, **MASKLID_BALANCED)
# MASKLID_BALANCED: beta=20, alpha=5, min_length=1, min_prob=0.3, assign="switch", switch_penalty=4, min_chars=6
# MASKLID_CAUTIOUS: the paper's beta=20, alpha=3, min_length=10, min_prob=0.9 with assign="switch": fewer false
#                   code-switches on monolingual text
# both presets set scripts=True (GlotScript >= 2.1)
```

With `scripts=True`, [GlotScript](https://github.com/cisnlp/GlotScript) gives the script of every word: a word only gets a
language written in its script (Han characters go to Chinese, Japanese or Korean labels, Latin words to Latin-script
labels), and a script that none of the found languages is written in gets a language of its own, predicted on that
script's words among labels of the script. A language that is the only one of its words' script is kept whatever its
length. On the test split below this raises the exact match on Chinese-English from 36.1 to 69.3 (CodeMixBench) and
from 58.9 to 60.9 (ASCEND), and leaves the subsets with one script unchanged.

Results on the test split of [cis-lmu/cs-lid-eval](https://huggingface.co/datasets/cis-lmu/cs-lid-eval) (30 subsets:
11 original code-switching corpora, CodeMixBench's 10 LID sets and ILP4LID's 9 sets with its monolingual controls),
GlotLID v4, settings chosen on its dev split; exact match (EM) of the language set and macro F1 of the word labels
(on the 18 subsets with word labels), macro averages over subsets:

| setting | EM all | EM code-switched | EM monolingual | word macro F1 | unassigned words |
|---|---|---|---|---|---|
| MaskLID, paper settings, 200 most frequent labels | 35.4 | 34.2 | 71.6 | 44.0 | 26.5% |
| `MASKLID_CAUTIOUS` without scripts | 39.8 | 38.1 | **88.7** | 57.4 | 1.7% |
| `MASKLID_BALANCED` without scripts | 46.3 | 45.2 | 77.7 | 63.5 | 1.7% |
| `MASKLID_BALANCED` (`scripts=True`) | **47.4** | **46.3** | 77.7 | | |

(Word macro F1 with `scripts=True` rises by 1.0 point over the same run without it; on Chinese-English from 61.9 to 79.7.)

### Numbers, punctuation and symbols

Characters of the Unicode categories numbers, punctuation and symbols are split off at the start and end of a word
(`"race car"` is `"`, `race`, `car`, `"`; `2024!` is `2024`, `!`) and never used to identify a language; inside a
word they stay (`it's`, `de3v`). With `return_words=True` they are returned with the label `None`.

### Changing the languages without reloading the model

`with_languages` returns a MaskLID limited to other languages that shares the loaded model (and its cache of word vectors) with the first one, so there is no need to load the model again:

```python
all_glotlid = MaskLID(model_path)                       # all labels
flores_model = all_glotlid.with_languages(flores_glotlid)
two_model = all_glotlid.with_languages(['__label__tur_Latn', '__label__eng_Latn'])
```

### Using an already loaded model, and labels per word

`MaskLID` also accepts a model loaded with `fasttext.load_model`, which it uses without copying, and `predict_codeswitch(..., return_words=True)` also returns the words and the label each was assigned to (None for words no language took):

```python
import fasttext
ft = fasttext.load_model(model_path)
masklid_model = MaskLID(ft)
ans, words, labels = masklid_model.predict_codeswitch(text, beta=20, alpha=3, max_lambda=3, min_length=10, min_prob=0.90, return_words=True)
```

### Probabilities over the selected languages

By default (as in the paper), `predict` takes the softmax over all labels of the model and then reads the selected languages from it. A text in none of the selected languages therefore gets low probabilities for all of them, and `min_prob` rejects it. With `renormalize=True`, the softmax is computed over the selected languages only, so their probabilities sum to 1 (contributed by [@jradola](https://github.com/jradola) in [#1](https://github.com/cisnlp/MaskLID/pull/1)):

```python
MaskLID(model_path, languages=flores_glotlid, renormalize=True)
# or: all_glotlid.with_languages(flores_glotlid, renormalize=True)
```

Use it when the text is known to be in one of the selected languages; keep the default when it may not be, since renormalized probabilities are high even for text in none of them.

## Citation

If you find our method and code useful for your research, please cite:

**ACL citation (preferred)**

```
@inproceedings{kargaran-etal-2024-masklid,
	title        = {MaskLID: Code-Switching Language Identification through Iterative Masking},
	author       = {Kargaran, Amir Hossein and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
	year         = 2024,
	booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
	publisher    = {Association for Computational Linguistics},
	address      = {Bangkok, Thailand}
}
```

**ArXiv citation**

```
@article{kargaran2024masklid,
  title={MaskLID: Code-Switching Language Identification through Iterative Masking},
  author={Kargaran, Amir Hossein and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2406.06263},
  year={2024}
}
```
