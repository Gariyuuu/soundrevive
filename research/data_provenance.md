# SoundRevive — Data Provenance Research

Research date: 2026-09-07. Methodology: web search (WebSearch tool) to locate primary source pages, followed by direct page fetch (WebFetch tool) to read the actual license/rights text where the page allowed automated fetching. Several primary pages (Library of Congress, UCSB Cylinder Audio Archive, Edinburgh DataShare) returned **HTTP 403 Forbidden** to the automated fetcher; for those, the license/rights claims below are based on search-engine result snippets that quote or closely paraphrase the primary page, and are flagged as such. Anyone building the benchmark should re-verify the 403'd pages by visiting them in a normal browser before finalizing redistribution decisions — this document is a research summary, not a legal opinion.

---

## Part 1 — Clean reference speech datasets (Regime A raw material)

### 1.1 LJSpeech

- **Primary source:** https://keithito.com/LJ-Speech-Dataset/
- **Verified via:** direct WebFetch of the page, 2026-09-07.
- **License:** Public domain, both audio and text. The page states the recordings (made 2016–17 by a LibriVox-affiliated project, single speaker "Linda Johnson") and the underlying texts (7 non-fiction books published 1884–1964) are in the public domain in the US. No attribution is legally required; the maintainer requests (not requires) a citation if used in a publication.
- **Sample rate:** 22,050 Hz, 16-bit mono PCM WAV.
- **Size:** 13,100 short clips, ~23 hours 55 minutes total.
- **Fit:** Single speaker, clean studio-like read speech, unambiguous public-domain status. This is the strongest candidate for the benchmark's clean-reference ground truth in Regime A, and for redistributable demo clips, because there is no attribution or licensing obligation at all.

### 1.2 LibriSpeech

- **Primary source:** https://www.openslr.org/12
- **Verified via:** direct WebFetch of the page, 2026-09-07.
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) — attribution required. Created by Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; derived from public-domain LibriVox audiobooks, segmented/aligned against Project Gutenberg text.
- **Sample rate:** 16 kHz.
- **Size:** ~1000 hours, multi-speaker read English speech.
- **Fit:** Good for scale/speaker diversity; usable for redistributable demo clips provided attribution to the corpus creators (and, transitively, LibriVox) is included. Lower sample rate (16 kHz) than LJSpeech, which matters for a benchmark that includes bandwidth-extension degradations — 16 kHz already caps the achievable "clean" bandwidth.

### 1.3 VCTK (CSTR VCTK Corpus, University of Edinburgh)

- **Primary source:** Edinburgh DataShare, https://datashare.ed.ac.uk/handle/10283/3443 (version 0.92); mirrored on HuggingFace (`CSTR-Edinburgh/vctk`) and elsewhere.
- **Verified via:** WebSearch only — direct WebFetch of both the Edinburgh DataShare collection page and the HuggingFace README returned **HTTP 403 Forbidden**. License claim is based on secondary sources: the HuggingFace dataset card, the `Real-Time-Voice-Cloning` GitHub repo's citation file, and the TensorFlow Datasets catalog entry, all of which independently state CC BY 4.0.
- **License:** Reported as CC BY 4.0 by multiple secondary sources. **UNVERIFIED against the primary DataShare license page** — could not fetch it directly in this session. Recommend a human visit https://datashare.ed.ac.uk/handle/10283/3443 directly to confirm before relying on this for redistribution.
- **Sample rate:** 48 kHz (higher than LJSpeech/LibriSpeech).
- **Size:** 110 speakers (various English accents), ~400 sentences each, ~44 hours.
- **Fit:** Higher native sample rate makes it attractive for bandwidth-extension-degradation tests, but the license is only secondarily confirmed — treat as a "probably safe, verify before shipping" source, not a "confirmed safe" one.

### 1.4 Mozilla Common Voice

- **Primary source:** https://commonvoice.mozilla.org/ ; dataset metadata repo https://github.com/common-voice/cv-dataset
- **Verified via:** WebSearch of the Mozilla Foundation's own blog post ("Common Voice 18 Dataset Release") and the `common-voice` GitHub org. Attempted direct fetch of `github.com/common-voice/common-voice/blob/main/LICENSE`, which returned the **software** license (MPL 2.0) — this is the code license, not the dataset license, and the page itself does not state the data license. The CC0 dataset-license claim rests on the Mozilla Foundation blog and multiple secondary aggregator pages (Hugging Face, Digital Public Goods), which is a good but not primary-page-confirmed source.
- **License:** Reported as CC0 1.0 ("no rights reserved") for the voice clips and transcripts. Note this is distinct from the MPL 2.0 license on the Common Voice web-app *code*.
- **Sample rate:** Variable — crowd-sourced recordings on consumer devices, not a fixed studio condition.
- **Size:** 30,000+ hours, 129+ languages (Common Voice 18 release, per Mozilla's own blog).
- **Fit:** CC0 is the least restrictive license of the four (no attribution obligation at all), but the crowd-sourced, variable-quality recording condition is a real confound if used as "clean" ground truth — individual clips would need quality filtering (background noise, clipping, mic quality) before use as a Regime-A clean reference. Good candidate for redistributable demo clips once filtered, less good as a scientific "clean" baseline without that filtering step.

**Recommendation for Regime A clean reference:** LJSpeech is the safest and cleanest choice (true public domain, single speaker, known clean recording condition). LibriSpeech is the best second choice for scale (CC BY 4.0, easy attribution). VCTK's higher sample rate is attractive but its license should be re-confirmed against the primary page before use. Common Voice is safest legally (CC0) but needs per-clip quality vetting before being treated as "clean."

---

## Part 2 — Public-domain / historical audio sources (Regime B raw material)

### 2.1 Internet Archive — general

- **Verified via:** WebSearch of Internet Archive Help Center ("Rights" page, https://help.archive.org/help/rights/) and forum threads.
- **What it actually says:** The Internet Archive tags many items with a `possible-copyright-status` metadata field (e.g., `NOT_IN_COPYRIGHT`), and this can be queried via the IA Search API. **Important caveat, stated by Internet Archive itself:** IA does **not guarantee** the copyright status of items hosted on archive.org and explicitly disclaims responsibility for the accuracy of copyright/rights metadata on item or collection pages. This means IA hosting a recording, or even tagging it `NOT_IN_COPYRIGHT`, is not by itself a legal guarantee — it is a curator's best-effort claim, not a warranty.
- **Recommendation:** Any specific IA item considered for redistribution needs its own item-level rights check (metadata field + item description), not a blanket assumption that "it's on archive.org so it's public domain."

### 2.2 LibriVox (public-domain audiobooks, useful as "historical-style" clean speech)

- **Primary source:** https://wiki.librivox.org/index.php/Copyright_and_Public_Domain
- **Verified via:** WebFetch attempted, returned HTTP 403; the content below is from WebSearch result snippets that directly quote the page.
- **What it actually says:** LibriVox only records texts already in the public domain in the US (generally: published before 1923, per the wiki's own framing — note LibriVox's site content on this predates the 2022 rolling public-domain-date changes and should be cross-checked per-recording against current US public domain dates rather than treated as a fixed cutoff). All LibriVox recordings themselves are released into the public domain — deliberately, not under a Creative Commons license — "so that people can do anything they like with them," including sell, broadcast, remix, or use commercially. No attribution is required (though appreciated).
- **Fit:** Genuinely excellent for the benchmark. LibriVox recordings are (a) real human speech performances (not studio-clean like LJSpeech, but not degraded originals either), (b) explicitly, unambiguously public domain by the hosting project's own stated policy, and (c) plentiful and diverse in speaker/reading style/recording quality — which is itself useful, since LibriVox home recordings have naturally varying period-appropriate-ish noise floors, room tone, etc. This can serve either as a second clean-ish reference corpus for Regime A, or (older, poorer-quality recordings) as informal Regime-B-adjacent material. **Recommend as one of the safe redistributable sources**, contingent on confirming the specific recording's public-domain status via LibriVox's own catalog metadata for the specific title used.

### 2.3 Library of Congress — National Jukebox

- **Primary source:** https://www.loc.gov/collections/national-jukebox/about-this-collection/rights-and-access/
- **Verified via:** WebFetch attempted, returned HTTP 403. Findings below are from WebSearch snippets that quote/paraphrase the LOC page and related LOC blog posts.
- **What it actually says (important nuance — LOC's terms are NOT simply "public domain, free to use"):**
  - Recordings were issued on labels now owned by Sony Music Entertainment. Sony granted LOC a **gratis license to stream** the acoustical recordings — this is a license LOC holds to stream them, not necessarily a statement that every recording is unrestricted for reuse by third parties.
  - At launch (2011), the Jukebox was **streaming-only, no downloads**.
  - Following the Music Modernization Act taking effect January 1, 2022, all US sound recordings published before 1923 entered the public domain. LOC's Recorded Sound section responded by beginning to enable **downloads for a select set** of recordings published 1900–1922 — not the entire catalog, and not automatically for everything that is technically now public domain.
  - The LOC rights-and-access page's general boilerplate (per search snippets) states you need written permission from rightsholders to copy/distribute copyrighted material except under fair use, and that other rights (publicity/privacy) may also apply — standard LOC across-the-board caution language.
  - Attribution is recommended but not strictly required.
- **Fit:** This is exactly the kind of "restrictive despite being technically public domain" case the user flagged. **Recommend: reference by metadata/link only, do not bulk-redistribute.** If a specific pre-1923 item is confirmed by LOC as one of the selected downloadable public-domain items, that single item could potentially be used for a demo clip citing LOC — but this needs per-item confirmation on loc.gov, not a general policy assumption, and the primary rights page could not be directly fetched in this session to confirm current wording.

### 2.4 UCSB Cylinder Audio Archive

- **Primary source:** https://cylinders.library.ucsb.edu/licensing.php
- **Verified via:** WebFetch attempted, returned HTTP 403. Findings below are from WebSearch snippets quoting the page and the archive's news page.
- **What it actually says:**
  - Cylinder recordings made **before December 31, 1922** are, as of January 1, 2022, in the public domain in the US, and UCSB explicitly states these can be **freely downloaded and used for any purpose, commercial or non-commercial**.
  - MP3 transfers of cylinders recorded **on or after January 1, 1923** are copyrighted by the Regents of the University of California (2005–2022) and licensed **CC BY-NC 2.5** (non-commercial only); commercial use of these requires a paid license from UCSB, and the required attribution credit line is "University of California, Santa Barbara Library."
  - Original WAV files (unedited or restored) of post-1922 cylinders can be requested for commercial or non-commercial use (CD reissues, film/TV sync, exhibits) — implying a separate licensing arrangement outside the free MP3 tier.
- **Fit:** This is the cleanest, most explicit public-domain historical-audio source found. Pre-1923 recordings (the large majority of the collection, since cylinders were made 1893–mid-1920s) are stated by UCSB itself, in plain language, to be free for any use including commercial. **Recommend as a safe redistributable source for Regime B demo clips**, restricted specifically to cylinders UCSB's own catalog dates to before 1923 — do not use post-1922 cylinders without checking the CC BY-NC restriction (which would forbid inclusion in a benchmark that might be used/cited commercially, or at minimum requires non-commercial framing + UCSB attribution).

### 2.5 Note on wax-cylinder AI restoration precedent

Not a data source per se, but relevant context found during this research: the Internet Archive ran an "AI Audio Challenge" (blog post, blog.archive.org, April 2023) specifically on restoring 78rpm records using expert-example training data, and multiple recent papers (Moliner & Välimäki and collaborators — see literature matrix) specifically target UCSB/IA-style cylinder and 78rpm gramophone material for denoising/bandwidth-extension. This confirms the benchmark's Regime B use case (no-reference restoration evaluation on real historical media) is an active, real research area, not a hypothetical one.

---

## Part 3 — Explicit recommendation

**Safe to build redistributable demo clips from (in order of confidence):**

1. **LJSpeech** — true public domain, verified via direct fetch of the primary page, no caveats.
2. **UCSB Cylinder Audio Archive, pre-1923 items only** — explicit "free for any use, commercial or non-commercial" statement from the rights-holder itself (per search-snippet quotation of their licensing page — recommend a human double-check of https://cylinders.library.ucsb.edu/licensing.php directly, since the automated fetch was blocked).
3. **LibriVox recordings** (as historical-style clean speech, or as a secondary Regime-A reference) — explicit, deliberate public-domain release policy stated by the project itself, contingent on confirming public-domain status of the specific recorded text/performance used.
4. **LibriSpeech** — CC BY 4.0, safe with attribution, verified via direct fetch.

**Reference by metadata/link only — do NOT redistribute clips:**

- **Library of Congress National Jukebox** — terms are genuinely restrictive/mixed despite post-2022 public-domain status for pre-1923 recordings; LOC has only enabled downloads for a curated subset, and the underlying streaming rights derive from a Sony Music gratis license rather than an unrestricted public grant. Treat any specific recording as needing its own per-item confirmation.
- **VCTK** — license (CC BY 4.0) is reported consistently by secondary sources but the primary Edinburgh DataShare license page could not be fetched directly in this session (403). Not "unsafe," but "unverified against the primary source" — worth a five-minute manual check before use.
- **Generic Internet Archive items** without an item-level rights-statement confirmation — IA itself disclaims responsibility for the accuracy of its copyright-status metadata, so "found on archive.org" is not sufficient evidence of clearance on its own.
- **Mozilla Common Voice** — legally the safest (CC0), but flagged separately here because its use as *historical-style* or *clean-reference* material would require quality filtering; it's a fine redistribution-safe source, just not automatically a fine "clean ground truth" source without that filtering step.

**Open gap:** no source found in this research qualifies as a large, unambiguously-clear, high-fidelity corpus of genuine early-20th-century *speech* (as opposed to music) recordings that is both (a) definitively public domain and (b) available as clean high-quality transfers with no non-commercial restriction. The UCSB cylinder collection is heavily music/vaudeville/political-speech-snippet oriented and not a large corpus of continuous historical speech; LOC's holdings include historical speech (e.g., early Roosevelt/Bryan/Taft recordings, per search results) but sit behind LOC's more restrictive terms. This gap is worth stating plainly in the benchmark's own documentation rather than glossing over: Regime B's genuine-historical-speech material will likely be UCSB pre-1923 cylinder speech recordings (small volume, but rights-clear) supplemented by LOC items used only for no-redistribution, reference-by-link evaluation.
