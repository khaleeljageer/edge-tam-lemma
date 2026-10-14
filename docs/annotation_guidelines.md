# Annotation guidelines: Tamil word → lemma

Used for the reviewed expansion batch (`data/annotation/wiki_batch01_reviewed.tsv`)
and the second-annotator agreement check (`data/annotation/agreement/`). The goal
is the lemma the UD Tamil treebanks would give, because the test set is scored
against them. Words are judged out of context, as single word forms.

## The lemma

| Word class | Lemma | Examples |
|---|---|---|
| Nouns | bare singular form: remove plural, case and clitic endings | மேற்கோள்கள் → மேற்கோள்; மனைவியும் → மனைவி; தாழ்வாரமோ → தாழ்வாரம் |
| Nouns with a changed stem | restore the dictionary form of the stem | விரோதச் → விரோதம்; அந்தஸ்தைப் → அந்தஸ்து; விடுதலைச் → விடுதலை |
| Verbs | bare verb root (the command form) | வாழ்வதோ, வாழவும் → வாழ்; உட்காருகின்றான் → உட்கார்; ஆவார் → ஆ |
| Proper names | remove case endings, otherwise keep the name | குமாருடையது → குமார்; நியூயார்க்கில் → நியூயார்க் |
| -ஆக adverbs | reduce to the noun | அழகாக → அழகு; உதவியாக → உதவி; உதாரணமாக → உதாரணம் |
| Fixed adverbs | keep as they are | அதிகமாக, குறிப்பாக, வேகமாக, எதிராக, மாறாக, மேலாக |
| Postpositions, numbers, conjunctions | keep as they are | அத்துடன், முப்பது, அங்கே |
| Pronouns | the treebank's base pronoun | தங்களின் → தன்; எங்களால் → எங்கள் |

Clitics -உம், -ஓ, -ஏ are always removed.

## Columns

- `lemma`: the lemma, in Tamil script.
- `pos` (optional): `NOUN`, `VERB` or `OTHER`.
- `note` (optional): anything worth flagging, e.g. "ambiguous: X or Y".

If a word is not a real Tamil word (a typo, broken text, a foreign word written in
Tamil script without a Tamil lemma), write `x` in `note` and leave `lemma` empty.
If unsure, leave `lemma` empty: empty rows are not counted.
