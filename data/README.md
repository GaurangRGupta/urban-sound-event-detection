# Data directory

This folder holds two public datasets. Nothing in this repository has been trained on them yet.

`raw/desed` is domestic sound-event detection with timestamps. `raw/fsd50k` is a large set of clip tags and has no timestamps. DESED is about 19.7 GB, most of it audio. FSD50K is about 41 MB of tables. The wave files that are on disk are evaluation clips. The official training waves are not in this folder.

## Layout

```
data/raw/
  desed/          domestic 10-second clips, 10 home sound classes
    archives/     the zip and tar.gz files that were downloaded
    audio/        real public-test waves
    metadata/     real-data label tables (train, validation, public test)
    dcase2019/    synthetic evaluation waves and their labels
    audioset_strong.tsv
    license_public_eval.tsv
  fsd50k/         51,197 Freesound clips, 200 classes, labels only
    archives/
    FSD50K.ground_truth/
    FSD50K.metadata/
    FSD50K.doc/
```

There is no `processed` folder and no field-recording folder yet. Two `.ipynb_checkpoints` folders are Jupyter autosave copies: a duplicate of `audioset_strong.tsv`, and 7 copied wavs under `desed/audio/eval/public/.ipynb_checkpoints`. They are not part of either dataset.

Every DESED clip is about 10 seconds at 44.1 kHz. A real filename such as `YKK227gPpRn4_30.000_40.000.wav` means seconds 30 to 40 of that YouTube video. Synthetic files are numbered, such as `1.wav`, and the matching label row says `1.jams`.

The 10 DESED class names, used in every strong label file, are:

`Alarm_bell_ringing`, `Blender`, `Cat`, `Dishes`, `Dog`, `Electric_shaver_toothbrush`, `Frying`, `Running_water`, `Speech`, `Vacuum_cleaner`.

## DESED: what each path holds

`desed/archives/` is the compressed download, about 8.2 GB.

- `6444477.zip` is the small Zenodo pack of real-data labels.
- `DESED_public_eval.tar.gz` (890 MB) unpacked into `audio/` and `metadata/`.
- `DESED_synth_eval_dcase2019.tar.gz` (7.7 GB) unpacked into `dcase2019/`.

Read the extracted folders. The archives are only there so the download can be checked or repeated.

`desed/audio/eval/public/` has 692 real wav files. This is the public test set. 616 files are stereo, 71 are mono, and 5 have 6 channels. Labels are in `metadata/eval/public.tsv`: 2,765 rows, and every wav name matches a label row.

A strong-label row has four columns.

| column | type | example | meaning |
| --- | --- | --- | --- |
| `filename` | string | `1Ro0FgMWTUE_120_130.wav` | which clip |
| `onset` | seconds, float | `0.000` | when the sound starts |
| `offset` | seconds, float | `2.081` | when it ends |
| `event_label` | string | `Speech` | which of the 10 classes |

One clip has several rows when several sounds occur. Speech from 0.0 to 2.1 seconds and Frying from 3.8 to 10.0 seconds are two rows for the same file.

`desed/license_public_eval.tsv` has 637 rows: `filename`, `uploader`, `uploader_id`, `license`. It covers 637 of the 692 public clips. It is attribution, not a learning target.

`desed/metadata/train/weak.tsv` is training labels for 1,578 real clips. Columns are `filename` and `event_labels`. The second column is a comma-separated list, for example `Alarm_bell_ringing,Speech`. It says those classes occur somewhere in the 10 seconds. It does not say when. The wav files for these clips are not on disk.

`desed/metadata/train/unlabel_in_domain.tsv` is 14,412 training filenames and nothing else. No class, no time, and no wav.

`desed/metadata/validation/validation.tsv` is the validation set: 1,168 clips and 4,239 strong-label rows, with the same four columns as the public test. The wav files are not on disk. The other two files in that folder are the same 1,168 clips split the way DCASE 2018 named them.

- `eval_dcase2018.tsv`: 880 clips, 3,343 events.
- `test_dcase2018.tsv`: 288 clips, 896 events.

Those two lists do not overlap, and together they are exactly `validation.tsv`. For this project they are validation. They are not a second test set.

`desed/audioset_strong.tsv` has 15,446 strong-label rows on 3,470 clips, with the same four columns. It is extra timestamp annotation for AudioSet clips in the DESED class list. The wavs are not here. 171 of these filenames also appear in the weak training list, 1,355 appear in the unlabeled list, and 35 appear in validation. It is additional annotation sitting on top of the real-data lists, not its own split.

`desed/dcase2019/dataset/` is synthetic evaluation only. Scaper mixed isolated home sounds on top of background noise, and the recipe kept the start and end times. There are 16 audio folders under `audio/eval/`, each with a tsv of the same name under `metadata/eval/`. The label column `filename` ends in `.jams`. The wav has the same number: `1.jams` belongs to `1.wav`.

| folder | clips | what changed |
| --- | --- | --- |
| `fbsnr_0dB`, `fbsnr_15dB`, `fbsnr_24dB`, `fbsnr_30dB` | 754 each | Same scenes and same timestamps. The foreground is quieter or louder against the background. 0 dB means the event is close to the noise. 30 dB means the event is much louder. 2,133 event rows. |
| `distorted_clipping`, `distorted_drc`, `distorted_highpass_filter`, `distorted_lowpass_filter`, `distorted_smartphone_playback`, `distorted_smartphone_recording` | 754 each | The 0 dB scenes again, after one degradation each. Same timestamps. |
| `500ms`, `5500ms`, `9500ms` | 750 each | One event per clip. The number is where the event starts: near 0.5 s, near 5.5 s, or near 9.5 s. |
| `ls_0dB`, `ls_15dB`, `ls_30dB` | 783 each | A long sound sits in the background and short sounds sit in the foreground, at three loudness ratios. 1,586 event rows. |

The same number in two of these folders is the same recipe id. Do not stack those folders into one count. `fbsnr_30dB` is the clean reference copy used in the audit notebook.

The synthetic training set is not in this tree. DCASE built a separate training mix: a couple of thousand 10-second files in the 2019 release, and about 10,000 in later releases. That archive is the Zenodo file `dcase_synth.zip`, and it was not downloaded.

## FSD50K: what each path holds

There are no wav files. A clip id such as `64760` would be `64760.wav` if the audio archives were downloaded. They were not.

`FSD50K.ground_truth/vocabulary.csv` has no header. Each of the 200 lines is `index, class name, AudioSet id`, for example `14,Bird,/m/015p6`.

`dev.csv` has 40,966 clips. Columns:

| column | type | example | meaning |
| --- | --- | --- | --- |
| `fname` | integer | `64760` | Freesound clip id |
| `labels` | comma-separated strings | `Electric_guitar,Guitar,...,Music` | classes present somewhere in the clip |
| `mids` | comma-separated ids | `/m/02sgy,/m/0342h,...` | the same classes as AudioSet ids |
| `split` | `train` or `val` | `train` | the official development split |

The label list includes parent names. A guitar clip is also tagged `Musical_instrument` and `Music`. That is why a count of the `Music` tag is very large.

The `split` column is the official cut: 36,796 `train` and 4,170 `val`.

`eval.csv` has 10,231 clips and the same columns except `split`. The whole file is the test set. Its labels were checked more carefully than the dev labels.

`FSD50K.metadata/collection/collection_dev.csv` and `collection_eval.csv` are the labels the annotators typed before parent names were added. The guitar clip there is just `Electric_guitar`. `vocabulary_collection_dev.csv` and `vocabulary_collection_eval.csv` are the class lists for those raw labels, again with no header.

`dev_clips_info_FSD50K.json` and `eval_clips_info_FSD50K.json` are one record per clip: title, description, tags, license URL, and uploader. They are keyed by the Freesound id.

`class_info_FSD50K.json` is annotator help for each of the 200 classes: a short FAQ and example descriptions. It is documentation, not a target.

`pp_pnp_ratings_FSD50K.json` stores, for each clip and each class id, the votes on whether that sound is the main sound in the clip (present and predominant) or only in the background (present but not predominant).

`FSD50K.doc/README.md` and `LICENSE-DATASET` describe the release. Each clip also has its own Creative Commons license, recorded in the clip-info JSON.

`fsd50k/archives/` holds the three small zips those folders were unpacked from.

## Train, validation, and test

The role below is the official role of each table. "Audio here" means the wav is in this `data/` folder today.

| set | role | clips | what the label gives you | audio here |
| --- | --- | --- | --- | --- |
| DESED `train/weak.tsv` | train | 1,578 | class list for the whole clip | no |
| DESED `train/unlabel_in_domain.tsv` | train pool | 14,412 | filename only | no |
| DESED `audioset_strong.tsv` | extra timestamps on real clips, mostly the train lists | 3,470 | start, end, class | no |
| DESED `validation/validation.tsv` | validation | 1,168 | start, end, class | no |
| DESED `eval/public.tsv` plus `audio/eval/public` | public test | 692 | start, end, class | yes |
| DESED `dcase2019/dataset` (all 16 folders) | synthetic test, several conditions of the same idea | 750 to 783 per folder | start, end, class | yes |
| FSD50K `dev.csv` where `split` is `train` | train | 36,796 | class list for the whole clip | no |
| FSD50K `dev.csv` where `split` is `val` | validation | 4,170 | class list for the whole clip | no |
| FSD50K `eval.csv` | test | 10,231 | class list for the whole clip | no |

These lists do not share clips with each other, except `audioset_strong.tsv`, which repeats some filenames from the weak list, the unlabeled list, and 35 validation clips. Public test, validation, and weak train are disjoint.

The synthetic training mix is the piece that would let a model learn timestamps from audio we are allowed to train on. It is not in this directory. The waves that are here, the public test and the synthetic eval, are the sets a finished model is scored on.

## Supervised learning

The learning problem is supervised sound-event detection. Supervised means each training example comes with the answer the model is supposed to produce. That answer is what these tables contain.

There are two grades of that answer.

Strong labels are the full answer for this project. For a clip, each sound has a class, a start time, and an end time. If the clip is cut into short frames, every frame has a yes or no for each of the 10 classes, and more than one class can be yes at the same time. Public test, validation, synthetic eval, and `audioset_strong.tsv` are strong. The synthetic files are strong because the mixer wrote down the times when it built the clip.

Weak labels are a partial answer. The table says which classes are in the clip and does not say when. DESED weak train and all of FSD50K are weak. A model can still be trained from that. The loss is on the clip as a whole. The timestamps it outputs are checked against the strong sets.

The unlabeled file is a list of training clips with no class and no time. It would matter only in a semi-supervised setup, and only after those waves are downloaded.

The audio that matches the training labels is the part that is not on disk yet.
