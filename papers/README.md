# Papers for this sound event detection project

Yes. The problem in this project is timestamped polyphonic sound event detection in domestic audio. That task has a published literature. The main line is Task 4 of the Detection and Classification of Acoustic Scenes and Events (DCASE) challenge, and the dataset that line introduced is the Domestic Environment Sound Event Detection (DESED) dataset.

This folder holds free copies of the papers that define that task, the data, the metrics, and the usual neural-network setup. A publisher PDF that is sold by the Institute of Electrical and Electronics Engineers (IEEE) was left out. Where a free author copy, a free preprint, or an open proceedings copy exists, that copy is the file on disk.

Every file below was retrieved on 6 October 2026. The date comes from the HTTP `Date` headers on the downloads (`Tue, 06 Oct 2026`).

The PDF files are listed in the project `.gitignore`. This markdown file is the piece that should stay in git, because it is the record of where each file came from. Do not commit the PDFs.

## How to check a file you already have

From the project directory, in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 papers\<file name>
```

The SHA-256 must match the value in the entry below. The size in bytes is also recorded. A file that starts with the characters `%PDF` is a PDF. A file that starts with `<!doctype` or `<html` is a web page and is not the paper.

## How to download the set again

1. Create `papers/` inside `neural_networks_and_deep_learning_project` if it is missing.
2. Save each URL in the entries below under the file name given there.
3. Run `Get-FileHash -Algorithm SHA256` on the new file and compare it with the checksum in that entry.
4. Two URLs need extra care, and the entries say so. The MDPI article page link returned `403 Forbidden`, so the download used the `mdpi-res.com` link. The HAL file link returns an HTML browser check to an automated request. Open that link in a browser and save the PDF. The IEEE Xplore links were not used.

A direct `curl` of an arXiv PDF works. Example:

```powershell
curl.exe -L --fail -o papers\cakir2017_crnn_polyphonic_sed.pdf https://arxiv.org/pdf/1702.06286.pdf
```

Use the URL printed in each entry, not a search-page link.

## What was left out on purpose

- IEEE Xplore PDFs for the conference and journal versions that are paid. The free copy is described in the entry, and the paid link is recorded so it is clear it was not fetched.
- The UrbanSound8K paper (Salamon, Jacoby, and Bello, 2014). That dataset is short clips with one label per clip. This project is timestamped multi-label detection, so that paper is a different task.
- The audio archives. They are already in `data/` and were not downloaded again. The records are listed at the end of this file.
- Sci-Hub and any other paywall bypass. They were not used.

## 1. Turpault, Serizel, Shah, and Salamon, 2019

Title: Sound Event Detection in Domestic Environments with Weakly Labeled Data and Soundscape Synthesis.

Authors: Nicolas Turpault, Romain Serizel, Ankit Shah, Justin Salamon.

Venue: DCASE 2019 Workshop, New York, 25-26 October 2019.

DOI printed on the PDF: 10.33682/006b-jx26.

Why it is here: this paper introduces DESED. It defines DCASE 2019 Task 4, which is the task this project follows: detect domestic sound events and give start and end times, using weak labels (a tag for the whole clip, with no times) together with strongly labeled synthetic clips (a start time and an end time for each event). The synthetic clips were mixed with Scaper.

Access: free workshop PDF. This is the proceedings file, not a draft.

URL used: http://archive.nyu.edu/bitstream/2451/60771/1/DCASE2019Workshop_Turpault_44.pdf

The HTTPS form of the NYU archive returned an HTML browser check, so the download used the HTTP URL above.

Also public, and not the file that was saved: the HAL draft at https://inria.hal.science/hal-02160855. That draft says the analysis was written before the evaluation period closed. The NYU file is the workshop paper.

Retrieved: 6 October 2026.

File: `papers/turpault2019_desed_dcase2019_task4.pdf`

Size: 561273 bytes.

SHA-256: `1e0c3a3152b25f94d7ed7a319e307a32ebc0cccec354714b5f41fd2722c7b346`

## 2. Serizel, Turpault, Eghbal-Zadeh, and Shah, 2018

Title: Large-Scale Weakly Labeled Semi-Supervised Sound Event Detection in Domestic Environments.

Authors: Romain Serizel, Nicolas Turpault, Hamid Eghbal-Zadeh, Ankit Parag Shah.

Venue: DCASE 2018 Workshop, Surrey, United Kingdom, 19-20 November 2018.

Preprint: arXiv:1807.10501, version 1, 27 July 2018. The PDF is stamped with that version.

Why it is here: DCASE 2019 Task 4 is the follow-up to this task. The 2018 paper sets the domestic YouTube clips, the weak labels, and the requirement that the system still output start and end times. DESED reuses part of this data.

Access: free arXiv PDF.

URL used: https://arxiv.org/pdf/1807.10501.pdf

Retrieved: 6 October 2026.

File: `papers/serizel2018_dcase2018_task4_weak_labels.pdf`

Size: 174586 bytes.

SHA-256: `685f73bd74e8b741bd942ba6539407702730a7b1fc75abf0ead50a95fcc232dd`

## 3. Serizel, Turpault, Shah, and Salamon, 2020

Title: Sound Event Detection in Synthetic Domestic Environments.

Authors: Romain Serizel, Nicolas Turpault, Ankit Shah, Justin Salamon.

Venue: IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2020, Barcelona, 4-8 May 2020, pages 86-90.

DOI: 10.1109/ICASSP40776.2020.9054478.

Why it is here: this is the analysis of DCASE 2019 Task 4 on synthetic soundscapes. The synthetic set is the kind of data this project trains and scores. The paper says time localization of events is still under-investigated when systems are trained from weak labels.

Access: the IEEE Xplore PDF is paid. Unpaywall, checked on 6 October 2026, listed that publisher version as closed and did not index a repository copy. The author manuscript is public on HAL, version 2, submitted 11 February 2020. That manuscript is the file on disk. The download includes the HAL cover page ("To cite this version") and then the manuscript. HAL's landing page lists the main file at about 381 KB. The saved file is larger because of that cover page.

Landing page: https://inria.hal.science/hal-02355573v2

URL used: https://inria.hal.science/hal-02355573/file/Sound_event_detection_in_domestic_environments_on_synthetic_soundscapes.pdf

An automated request to that URL receives an HTML browser check (HTTP 200, page title "Making sure you're not a bot!"), not the PDF. The PDF in this folder was saved from that same URL on 6 October 2026 after the public check. To get it again, open the URL in a browser and save the PDF, then compare the SHA-256. The same file path is the one listed by the HAL API as `files_s` for `hal-02355573`.

Paid link that was not downloaded: https://ieeexplore.ieee.org/document/9054478/

Retrieved: 6 October 2026.

File: `papers/serizel2020_sed_synthetic_domestic_environments.pdf`

Size: 475790 bytes.

SHA-256: `00ce17c7e16a5103b55d8aeee07b008fee889c3a6d213c3b45545ec444afc11c`

## 4. Turpault and others, 2021

Title: Sound Event Detection and Separation: A Benchmark on Desed Synthetic Soundscapes.

Authors: Nicolas Turpault, Romain Serizel, Scott Wisdom, Hakan Erdogan, John R. Hershey, Eduardo Fonseca, Prem Seetharaman, Justin Salamon.

Venue: ICASSP 2021, Toronto, 6-11 June 2021, pages 840-844.

DOI: 10.1109/ICASSP39728.2021.9414789.

Why it is here: the paper evaluates systems on synthetic DESED soundscapes and says localization in time is still a problem. Its event-based F-score uses a 200 ms collar on the onset and an offset collar of the larger of 200 ms and 20 percent of the reference event length. F-score here means the harmonic mean of precision and recall. This project uses that same collar.

Access: the IEEE Xplore PDF is paid. The file on disk is the arXiv preprint.

URL used: https://arxiv.org/pdf/2011.00801.pdf

That URL serves arXiv:2011.00801, version 1, dated 2 November 2020. The introduction in that file says the systems are the submissions to DCASE 2020 Task 4. One sentence in the abstract says DCASE 2021. Both sentences are in the file that was saved.

Paid link that was not downloaded: https://ieeexplore.ieee.org/document/9414789/

A HAL document URL for `hal-02984675` returned the same HTML browser check and was not kept.

Retrieved: 6 October 2026.

File: `papers/turpault2021_desed_synthetic_benchmark.pdf`

Size: 537314 bytes.

SHA-256: `5b3f4975ce76fa9713463766e5ae87e5db8db9b7118680903becb684579aad56`

## 5. Salamon, MacConnell, Cartwright, Li, and Bello, 2017

Title: Scaper: A Library for Soundscape Synthesis and Augmentation.

Authors: Justin Salamon, Duncan MacConnell, Mark Cartwright, Peter Li, Juan Pablo Bello.

Venue: IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, New York, 15-18 October 2017.

DOI: 10.1109/WASPAA.2017.8170052. This is the identifier used for Scaper in the reference list of Serizel, Turpault, Shah, and Salamon, 2020.

Why it is here: Scaper is the library that mixed the strongly labeled synthetic clips in DESED. A specification draws event start times, durations, and signal-to-noise ratio, and the library writes the mixture plus the strong labels.

Access: the IEEE Xplore PDF is paid. The file on disk is the copy in the U.S. National Science Foundation Public Access Repository.

URL used: https://par.nsf.gov/servlets/purl/10074705

Paid link that was not downloaded: the IEEE DOI above.

Retrieved: 6 October 2026.

File: `papers/salamon2017_scaper.pdf`

Size: 190167 bytes.

SHA-256: `744353532e289d582334570777944c45e23e3cebe0f06d7fd88b3c8f1f2d43cb`

## 6. Gemmeke and others, 2017

Title: Audio Set: An Ontology and Human-Labeled Dataset for Audio Events.

Authors: Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, Marvin Ritter.

Venue: ICASSP 2017, New Orleans, 5-9 March 2017, pages 776-780.

DOI: 10.1109/ICASSP.2017.7952261.

Why it is here: the real domestic clips in the DCASE task are YouTube excerpts labeled from the Audio Set ontology. The paper describes the 10 second clips and the weak (clip-level) labels. It also states the goal of recognizing events at a time resolution better than one second, which is the detection problem rather than one label for a whole clip.

Access: free PDF hosted by Google. The IEEE Xplore PDF was not downloaded.

URL used: https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/45857.pdf

Paid or publisher link that was not downloaded: https://ieeexplore.ieee.org/document/7952261/

Retrieved: 6 October 2026.

File: `papers/gemmeke2017_audioset.pdf`

Size: 1370291 bytes.

SHA-256: `afe56cf887db844ebff99f30a28be8e9088a02052a62e352fb0c429c689a7f53`

## 7. Fonseca, Favory, Pons, Font, and Serra, 2022

Title: FSD50K: An Open Dataset of Human-Labeled Sound Events.

Authors: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra.

Venue: IEEE/ACM Transactions on Audio, Speech, and Language Processing, volume 30, 2022, pages 829-852.

DOI: 10.1109/TASLP.2021.3133208.

The saved PDF is stamped arXiv:2010.00475, version 2, 23 April 2022, and it is typeset as the journal article.

Why it is here: FSD50K is the secondary dataset in this project. It is clip-level labels over 200 Audio Set classes, with the waveforms released under Creative Commons licenses. This project uses the label tables. It does not use FSD50K audio. The paper's own DOI for the data release is 10.5281/zenodo.4060432.

Access: free arXiv PDF. The IEEE Xplore PDF was not downloaded.

URL used: https://arxiv.org/pdf/2010.00475.pdf

Paid link that was not downloaded: https://ieeexplore.ieee.org/document/9645159/

Retrieved: 6 October 2026.

File: `papers/fonseca2022_fsd50k.pdf`

Size: 2955209 bytes.

SHA-256: `6187714fdf9f10097a74b6cce13c0849898cb7538fa46e3aae61ae1162018023`

## 8. Mesaros, Heittola, and Virtanen, 2016

Title: Metrics for Polyphonic Sound Event Detection.

Authors: Annamaria Mesaros, Toni Heittola, Tuomas Virtanen.

Venue: Applied Sciences, volume 6, issue 6, article 162, published 25 May 2016.

DOI: 10.3390/app6060162.

Why it is here: this paper defines the segment-based and event-based precision, recall, and F-measure used for polyphonic detection, where several events may be active at once. Segment-based scoring cuts the clip into fixed time bins. Event-based scoring matches a predicted event to a reference event when the onset and the offset fall inside a collar. This project uses both, with a 1 second segment and the 200 ms / 20 percent collar.

Access: open-access publisher PDF.

URL that failed: https://www.mdpi.com/2076-3417/6/6/162/pdf returned 403 Forbidden.

URL used: https://mdpi-res.com/d_attachment/applsci/applsci-06-00162/article_deploy/applsci-06-00162.pdf

Article page: https://www.mdpi.com/2076-3417/6/6/162

Retrieved: 6 October 2026.

File: `papers/mesaros2016_metrics_polyphonic_sed.pdf`

Size: 666413 bytes.

SHA-256: `02020ba884a20aa205712ed0b172ae153dbc26040460624cfd9b89b674d5046c`

## 9. Bilen, Ferroni, Tuveri, Azcarreta, and Krstulovic, 2020

Title: A Framework for the Robust Evaluation of Sound Event Detection.

Authors: Cagdas Bilen, Giacomo Ferroni, Francesco Tuveri, Juan Azcarreta, Sacha Krstulovic.

Venue: ICASSP 2020, Barcelona, 4-8 May 2020, pages 61-65.

DOI: 10.1109/ICASSP40776.2020.9052995.

Preprint: arXiv:1910.08440, version 2, 14 February 2020. The saved PDF carries that stamp and the IEEE author notice that personal use of the preprint is permitted.

Why it is here: later DCASE Task 4 papers report the polyphonic sound detection score (PSDS) from this paper. PSDS scores a system across many thresholds and relaxes the collar, because a single threshold and a tight collar mix operating-point choice with localization error. This project still reports collar-based segment and event F-measures, at the fixed threshold 0.5. The paper is here so those DCASE numbers can be read against a metric this project does not compute.

Access: free arXiv PDF. The IEEE Xplore PDF was not downloaded.

URL used: https://arxiv.org/pdf/1910.08440.pdf

Paid link that was not downloaded: https://ieeexplore.ieee.org/document/9052995/

Retrieved: 6 October 2026.

File: `papers/bilen2020_psds_evaluation.pdf`

Size: 666363 bytes.

SHA-256: `74ceb79095c355d2ab310cb60240e1c4464c27c63a319b9ce434a24becca0610`

## 10. Cakir, Parascandolo, Heittola, Huttunen, and Virtanen, 2017

Title: Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection.

Authors: Emre Cakir, Giambattista Parascandolo, Toni Heittola, Heikki Huttunen, Tuomas Virtanen. The preprint says Parascandolo and Cakir contributed equally.

Venue: IEEE/ACM Transactions on Audio, Speech, and Language Processing, volume 25, issue 6, June 2017, pages 1291-1303.

DOI: 10.1109/TASLP.2017.2690575.

Preprint: arXiv:1702.06286, version 1, 21 February 2017.

Why it is here: this is the standard neural-network paper for polyphonic sound event detection. It compares a convolutional recurrent neural network with a convolutional network, a recurrent network, and older classifiers. The recurrent part is the relevant comparison for the frame gated recurrent unit (GRU) in this project. A GRU is a small recurrent network that keeps a state across frames. The convolutional front end treats a mel spectrogram like an image. This project does not use that front end.

Access: free arXiv PDF. The journal PDF is paid and was not downloaded.

URL used: https://arxiv.org/pdf/1702.06286.pdf

Paid copy that was not downloaded: the IEEE version at DOI 10.1109/TASLP.2017.2690575. The download used arXiv only.

Retrieved: 6 October 2026.

File: `papers/cakir2017_crnn_polyphonic_sed.pdf`

Size: 4350445 bytes.

SHA-256: `76242309fdcf8e431d02693b2e6b4c97c4caad39ed38353b5720c76470552ad5`

## Data records already in the project

These are not papers and they were not downloaded again.

- DESED project page, printed in the 2020 Serizel manuscript: https://project.inria.fr/desed/
- DCASE challenge site: https://dcase.community/
- FSD50K data DOI, printed in the Fonseca PDF: https://doi.org/10.5281/zenodo.4060432 which resolves to https://zenodo.org/records/4060432
- The synthetic DESED audio already stored under `data/raw/desed/` in this project was taken from Zenodo record 6026841 (`dcase_synth`). The zip was not fetched again for this literature pass.

## Checksums in one block

```text
bilen2020_psds_evaluation.pdf 666363 74ceb79095c355d2ab310cb60240e1c4464c27c63a319b9ce434a24becca0610
cakir2017_crnn_polyphonic_sed.pdf 4350445 76242309fdcf8e431d02693b2e6b4c97c4caad39ed38353b5720c76470552ad5
fonseca2022_fsd50k.pdf 2955209 6187714fdf9f10097a74b6cce13c0849898cb7538fa46e3aae61ae1162018023
gemmeke2017_audioset.pdf 1370291 afe56cf887db844ebff99f30a28be8e9088a02052a62e352fb0c429c689a7f53
mesaros2016_metrics_polyphonic_sed.pdf 666413 02020ba884a20aa205712ed0b172ae153dbc26040460624cfd9b89b674d5046c
salamon2017_scaper.pdf 190167 744353532e289d582334570777944c45e23e3cebe0f06d7fd88b3c8f1f2d43cb
serizel2018_dcase2018_task4_weak_labels.pdf 174586 685f73bd74e8b741bd942ba6539407702730a7b1fc75abf0ead50a95fcc232dd
serizel2020_sed_synthetic_domestic_environments.pdf 475790 00ce17c7e16a5103b55d8aeee07b008fee889c3a6d213c3b45545ec444afc11c
turpault2019_desed_dcase2019_task4.pdf 561273 1e0c3a3152b25f94d7ed7a319e307a32ebc0cccec354714b5f41fd2722c7b346
turpault2021_desed_synthetic_benchmark.pdf 537314 5b3f4975ce76fa9713463766e5ae87e5db8db9b7118680903becb684579aad56
```
