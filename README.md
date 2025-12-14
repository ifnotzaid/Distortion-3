🇷🇺 Russian Speech Recognition: Benchmarking Architectures on the Golos Dataset
A comparative study of Automatic Speech Recognition (ASR) architectures—ranging from traditional CRNNs to modern Transformers—evaluated on the Russian SberDevices Golos dataset.

📊 The Dataset: Golos
This project utilizes the Golos dataset (Crowd Split), an open-source corpus of Russian speech provided by SberDevices.

Source: Crowdsourced recordings (people reading prompts via mobile phones).

Domain: Voice assistant commands, search queries, and general requests.

Language: Russian (Cyrillic).

Audio Format: Single-channel, varying sample rates (resampled to 16kHz for training).

Size:

Total: ~1,240 hours.

Used for Baseline: ~20,000 samples (Training), 100 samples (Evaluation).

Preprocessing Pipeline
To ensure fair comparison across models, the data was normalized as follows:

Resampling: All audio converted to 16,000 Hz.

Text Normalization:

Lowercasing.

Removal of punctuation (.,?!-).

Vocabulary restricted to the 33 Russian Cyrillic letters + Space + Blank.
