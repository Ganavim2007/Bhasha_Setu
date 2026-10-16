# Bhasha Setu

Bhasha Setu is an offline-first classroom prototype that helps teachers prepare Hindi or English curriculum in students' mother tongues. The configured local server generates Hindi-to-Santali drafts using the model's Bengali-script Santali tag (`sat_Beng`); it is not an Ol Chiki model. English-to-Santali goes through an experimental Hindi bridge. Santali document PDFs become available to students when translation finishes and are clearly marked as AI drafts until a reviewer approves every section. Santali text review requires one reviewer by default; set `BS_REQUIRED_REVIEWERS=2` to require both teacher and community-expert roles.

## Working in this prototype

- An installable PWA caches the app shell. Approved lessons can be stored on each device for offline study; the student view refreshes on reconnect, when the app returns to the foreground, and once a minute while visible.
- The student landing page is the first screen. Students can search approved lesson text, save lessons on the device, and open saved PDF sections without seeing the same section twice.
- Student view offers Santali, Ho, and Mundari. Ho and Mundari lessons are community-translated entries and appear only after both reviewer roles approve the same Hindi text, translation, writing system, and source note. Their approved text is cached locally for offline study and refreshed when connected.
- Once opened online, the Ho/Mundari lookup caches the reviewed word list locally and supports offline search. It syncs when opened again or while the app remains visible.
- Reviewers can submit Hindi-to-Ho and Hindi-to-Mundari curriculum text for human translation or contribute a completed pair. A teacher and a community language expert must approve the exact sentence, writing system, and source note before it appears in Student view; approved lessons can be returned for correction and are exported with review provenance.
- Signed-in reviewers can upload PDF, DOCX, PPTX, and UTF-8 TXT files up to 20 MB. PDF/PPTX inputs are limited to 120 pages/slides and extracted material to 2,000 review sections. PDF, Word, and PowerPoint text is read in source order; original page styling, images, and tables are not retained.
- Hindi source files can be sent to Santali, Ho, or Mundari. Santali gets a Hindi-to-Santali model draft; Ho and Mundari enter fluent-speaker translation tasks. English source files currently use the Santali path only: the installed multilingual model first creates a Hindi bridge, then drafts Santali. Reviewers see the English original and Hindi bridge. This English bridge has not been evaluated for curriculum accuracy, and every draft requires human correction or approval by the configured reviewer count.
- Each source page/slide or document section is kept as a separate review item with its source order. As soon as all Santali sections have AI drafts, students get one generated, continuously flowing PDF clearly marked as unreviewed. Each original Hindi or English section is followed by its paired Santali translation. After every Santali item passes the configured reviewer count, the student library offers the reviewed PDF. Ho and Mundari full documents remain unavailable until human translation and review are complete. PDFs do not insert a page break for each source page. PyMuPDF lays out the text and shapes complex scripts; images and table layout are not reproduced, while extracted table text flows as regular text.
- If a connection drops during upload, the selected file is saved in the browser's IndexedDB outbox with its source and target language. It retries when the device reconnects and a reviewer session is active. Browsers with Background Sync can wake the open app to start that retry.
- Review uses separate teacher and community-expert passcodes. Reviewers submit translations directly. By default, one signed-in reviewer can approve a Santali lesson for students; `BS_REQUIRED_REVIEWERS=2` restores the two-role check. Ho and Mundari sentence and word-pair content still requires both roles. Study-material cards and review sentences have individual Remove controls; removing a material cancels its queued translation, and removing a sentence removes it from linked document drafts.
- A signed-in reviewer can flag a published translation from Student view. It disappears from student lessons and saved packs on the next sync, then returns to the configured Santali review flow with the correction reason and requesting role visible. The reason, requesting role, review round, and request time are included in the approved review JSONL export. The reason stays with that review cycle even if an edit starts another round; the lesson is republished after the configured reviewer count approves the same text again.
- The review page renders a rolling batch of at most 20 drafts and shows the total waiting. Study-material progress shows translation and review status; the second-review count appears when two Santali reviewers are configured.
- Offline lesson packs include approved translations, recordings, and visuals. When the device syncs, saved sections update to the latest approved translation; sections that are no longer approved are removed. Changed recordings and visuals are fetched again when available. Packs from older approval formats are not shown.
- Drafts with lower model scores are shown first. The score is a review signal, not a calibrated probability of correctness.
- Short lesson read-aloud uses a matching Hindi, Santali, Ho, or Mundari voice installed on the student's device when available. Voice availability varies by device. Approved Santali lessons can also use locally generated speech from the optional Santali TTS model.
- The student-facing **Audio** section lets students upload a Hindi or English PDF for Santali translation, then listen to its sections in order. It plays approved Santali recordings when every section has one; otherwise text playback requires an installed Santali voice. Bengali-script Santali is still spoken Santali, so a Bengali voice is not used as a substitute. Text is saved in IndexedDB on that device after first opening it; approved recordings can be cached for offline playback.
- After the configured number of reviewers approve a Santali lesson, one reviewer can generate an AI speech draft on the school server, record it in a secure browser, or upload a recording. A reviewer listens through the entire clip and approves it before students can hear it. Generated clips are labeled as AI speech; human recordings stay on the school server. Each device caches approved audio after playback. Ho and Mundari TTS are not included.
- A reviewer can attach a school-approved still image to an approved lesson with a Santali accessible description and source credit. The other role must inspect and approve the exact image and description before students see it. Approved images can be saved in PDF lesson packs for offline study; uploads are resized, metadata is stripped, and files are stored under `lesson_visuals/`, which Git ignores. If the translation changes, the image must be reviewed again and the recording must be made again.
- Ho and Mundari primer entries imported through OCR start in a private review queue. Word lookup shows only entries checked by both the teacher and community language expert roles. Reviewers can correct an OCR pair, add a new pair, reject a bad source entry, or reopen a published pair for correction. Reviewed pairs can be exported with source and reviewer provenance; they remain separate from the Santali sentence-training dataset.
- Students can increase or reduce text size and switch on high contrast. These settings are saved on the device.
- Twice-reviewed translations can be exported as JSONL with both reviewer roles and edit information for a separately checked training process.
- A signed-in teacher or community expert can download a school backup ZIP from **Progress**. It contains consistent snapshots of both SQLite databases plus stored lesson audio and visuals; it excludes model weights, secrets, staged PDFs, and device-only offline caches.

## Back up and restore school data

Download **Progress → Download school backup** regularly and copy the ZIP to a secure location separate from the school server. The archive contains `bhasha_setu.db`, `bhasha_setu_v2.db`, `audio_recordings/`, `lesson_visuals/`, and a `backup_manifest.json`. The ZIP is not encrypted; it contains lesson text, review history, and media, so restrict access to it. Model weights and generated full-document PDFs can be rebuilt; passcodes must be configured in the server environment rather than restored from a backup.

To restore onto the same or a replacement server, stop Bhasha Setu with Ctrl+C first. Keep a separate copy of the current databases and media folders, then extract the backup ZIP into the project folder and replace the two database files and the `audio_recordings` and `lesson_visuals` folders. Restart using the school's configured passcodes. Reinstall optional dependencies and restore TLS settings separately if this is a replacement computer. Source upload files and the generated PDF cache are not included; upload source files again if needed. Generated student PDFs are recreated from the saved draft or approved text.

## Upload and publish a complete translated document

Sign in to **Study materials**, select the language of the file and the target language, then choose a PDF, DOCX, PPTX, or UTF-8 TXT file. Hindi-to-Santali receives a machine draft. English-to-Santali first uses the model's English-to-Hindi bridge and then the Hindi-to-Santali draft; the original English and bridge Hindi are available to reviewers. English upload is limited to Santali because the current Ho/Mundari workflow requires a Hindi source and fluent-speaker translation.

The server extracts source text in order and splits it into reviewable sections. Once the complete Santali model draft is ready, it appears in **Learn → Complete translated documents** as an **AI draft - review pending** PDF, so students can open and read it while reviewers check the text. In the PDF, each original Hindi or English section is followed by its corresponding Santali translation. The generated PDF keeps section order and flows continuously across pages. After every section is corrected as needed and the configured Santali reviewer count approves it, the student library offers the reviewed PDF. Ho and Mundari PDFs require human translation and both-role approval first. PDFs do not preserve original page breaks, images, table layout, fonts, or slide layout; extracted table text flows as regular text. Open each student PDF once while connected so the browser can cache it for offline reading on that device. Browser storage can be cleared by the device or browser, so it is not a school-server backup.

On the school server, install the document dependencies with `python -m pip install -r requirements.txt`, then restart `python api.py`. Scanned PDFs require the optional OCR packages and Tesseract. Install Hindi (`hin`) data for Hindi source scans or English (`eng`) data for English source scans. OCR output must be compared with the source before review.

Students can also open **Audio** and submit a PDF directly. These uploads use the same local extraction, Santali draft, teacher review, and publication process as Study materials. Audio labels an AI draft while review is pending. Student uploads are PDF-only, limited to 20 MB and 120 pages, and limited to three uploads per network address in 15 minutes. Translated text is cached on the student's device after they open it. Approved section recordings are played directly; sections without recordings need an installed Santali speech voice.

## Collect Ho and Mundari lessons

Sign in and open **Study materials**. Choose Ho or Mundari and enter the class's confirmed writing system before uploading Hindi source material. Extracted sections enter the **Ho/Mundari sentences** human translation queue with source page/section references. A fluent reviewer fills in each target translation; the teacher and community language expert then approve the exact version. Individual approved lessons appear in Student view. A whole-document PDF is listed only after every section is approved. An authorized reviewer can return an approved lesson for correction; the full document disappears from the student list until all sections pass review again.

Download a language-specific JSONL file from that page. Each row records the target language, writing system, source note, contributor role, reviewer roles and review timestamps. The app does not generate Ho or Mundari translations: human-produced, dual-approved pairs can be shown as lessons and can support future model development. They are not mixed into the Santali training corpus. Review the language data, orthography, licensing, and held-out evaluation with fluent speakers before training or classroom use.

## Record and review lesson audio

After the configured Santali text review is complete, sign in and open **Audio**. For Santali, choose **Generate Santali audio** on a lesson to create an AI speech draft from its approved text. The first generation needs the optional speech dependencies, accepted access to [AI4Bharat Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts), a Hugging Face read token, and an internet connection to download several GB of model weights. The model is cached on the school server and later generation can run offline. CPU-only generation may take several minutes and needs substantial memory. Santali speech pronunciation has not been independently evaluated, so generated clips stay drafts until a fluent reviewer listens and approves them. Only approved clips appear in Student view.

On Windows, stop the server with Ctrl+C, install the optional runtime in the project's active virtual environment, accept the model access terms in a browser, and set a read token in the same PowerShell window before restarting:

```powershell
python -m pip install -r requirements-tts.txt
$env:HF_TOKEN = "hf_your_read_token"
python api.py
```

Keep the token private; it is read from the environment and should not be committed. The first **Generate Santali audio** request downloads the speech model. Later requests use the local model cache. The reviewer who starts generation can preview and withdraw the draft under **Your Santali recordings**; a different role checks the clip before it is published. Browser microphone access for human recordings requires HTTPS or localhost; file upload remains available for WebM, Ogg, WAV, MP4, or MP3 clips. Audio files are stored in `audio_recordings/`, which Git ignores. Ho and Mundari clips still need fluent-speaker recordings; the app does not synthesize those languages.

Student devices cache approved audio when it is played online. When saving approved Santali PDF sections for offline study, the app also fetches the available clips for that pack.

## Language and model limits

The bundled LoRA adapter supports Hindi-to-Santali drafts. Ho and Mundari have reviewed word lookup lists and human-translated lesson workflows, not machine translation models. Do not present a word lookup or an unreviewed model output as a curriculum translation. Build and evaluate substantial, fluent-speaker-checked parallel corpora before enabling Ho or Mundari machine translation.

The [NLLB base model](https://huggingface.co/facebook/nllb-200-distilled-600M) is published under CC-BY-NC-4.0. Check its current license and attribution terms before redistributing model weights or using the model commercially.

The app does not start back-translation or training automatically. One command creates unvalidated reverse-translation candidates from checked Santali text:

```powershell
python training/backtranslate_santali.py .\reviewed_santali.txt --output .\training\synthetic_candidates.jsonl
```

The script uses the NLLB base model to generate Hindi from Santali and writes provenance fields with `synthetic: true` and `human_validated: false`. Review every pair before training; the training script rejects these rows until they are explicitly marked as checked by both roles. Reviewed corrections can train a separate candidate adapter with a distinct holdout set:

```powershell
python -m pip install -r requirements-training.txt
python training/train_santali_adapter.py --approved .\approved_pairs.jsonl --evaluation .\holdout_pairs.jsonl --output .\training\candidate-santali
```

The candidate is saved separately and compared with both the base model and the current adapter using chrF. See [training/README.md](training/README.md) for dataset requirements and review steps. It is not installed automatically. The current Santali adapter uses the Bengali-script Santali tag; agree on and validate the target writing system with the community before classroom use.

Text-based PDFs need no OCR setup. Scanned Hindi or English pages can use optional local Tesseract OCR; install Hindi (`hin`) data for Hindi source or English (`eng`) data for English source. OCR may misread letters; reviewers should check extracted text against the source. Santali whole-document drafts are student-accessible while review is pending and are labeled as AI drafts. The Santali adapter produces Bengali-script text; the optional English-to-Hindi bridge is experimental and has no classroom accuracy benchmark. Optional speech generation uses a separate TTS model. Generated audio is synthetic and requires reviewer listening. The app can use matching Hindi, Santali, Ho, or Mundari voices installed in the student's browser for local read-aloud; availability varies by device, and it will not substitute a different language voice. Automatic illustrations, quizzes, and individual accounts are not included. Uploads remain in the browser outbox until upload. The server stages the original file locally only until text extraction finishes, then removes it and keeps the filename, source text, and page/slide/section provenance.

## Optional OCR for scanned Hindi or English PDFs

Install the Python rendering and OCR wrapper on the school server:

```powershell
python -m pip install -r requirements-ocr.txt
```

Install the Tesseract OCR program separately from the [UB Mannheim Windows builds](https://github.com/UB-Mannheim/tesseract/wiki) and include Hindi (`hin`) and/or English (`eng`) language data, depending on the scanned files you upload. Confirm `tesseract --list-langs` includes the needed language code. If Windows does not find it on `PATH`, set the executable path before starting the server:

```powershell
$env:BS_TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
python api.py
```

OCR runs on the school server without sending page images to a cloud service. It is used for PDF pages that have no selectable text. The material view reports how many pages went through OCR. Check extracted Hindi or English against the scanned page; do not treat OCR output as verified source text.

## Run on Windows

From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:BS_TEACHER_PIN = "choose-a-private-teacher-passcode"
$env:BS_EXPERT_PIN = "choose-a-different-expert-passcode"
$env:BS_REQUIRED_REVIEWERS = "1"  # Use "2" to require both roles for Santali text
python api.py
```

Open <http://localhost:8000>. The passcodes must each have at least eight characters and must be different. Keep the same PowerShell window open while the server runs. The first model start may download the multilingual base model; the school server then needs no internet for inference if its model files are available locally. Existing PDF jobs in `bhasha_setu.db` resume after restart.

The server does not hot-reload Python changes. After updating the project, stop the old server with Ctrl+C in its PowerShell window and run `python api.py` again there, then reload the browser page. The app warns when the page and server versions do not support the same Ho/Mundari lesson features.

Both passcodes remain configured because the app also uses the teacher and community-expert roles for Ho/Mundari data and media review. Either role can approve a Santali lesson when `BS_REQUIRED_REVIEWERS=1` (the default). Set `$env:BS_REQUIRED_REVIEWERS = "2"` in PowerShell before `python api.py` to require both roles. Keep the passcodes private. This shared-passcode setup is for a supervised pilot, not a production identity system.

## Classroom Wi-Fi

Student devices can study downloaded lessons offline. Uploading PDFs and creating new translations needs a connection to the school server. The PWA requires HTTPS when accessed from another device. Configure a trusted school TLS certificate and private key, then start the server with:

```powershell
$env:BS_TEACHER_PIN = "choose-a-private-teacher-passcode"
$env:BS_EXPERT_PIN = "choose-a-different-expert-passcode"
$env:BS_HOST = "0.0.0.0"
$env:BS_SSL_CERTFILE = "C:\path\to\school-cert.pem"
$env:BS_SSL_KEYFILE = "C:\path\to\school-key.pem"
python api.py
```

Allow port 8000 only on the school's private Wi-Fi and open `https://<school-server-name>:8000` on student devices. The certificate must be trusted by those devices. Do not expose the classroom server to a public network.

The current adapter's training corpus and evaluation report are not included, so its original score is not reproducible. The training command records new data fingerprints and scores but cannot prove the older model never saw the holdout set. AI output can be wrong even when its score is high; check translations with fluent Santali speakers before classroom use.
