# Speaker Diarization
### Objective
The project is meant to handle the speaker diarization task; i.e. the task of determining which parts of a speech stream is uttered by each speaker. The main objective is to improve the current speaker diarization accuracy by investigating appropriate approaches. 
leading to the construction of a novel speech representation which holds the greatest speaker discrimination which is expected to improve the task at hand.
One of the most important application will be in transcription of conversation. It can be used to localise the instances of speaker instances to pool data for model training which in turn improve transcription accuracy. It can be used to crop the speech of a specific person of interest from a long audio clip.

### Methodology

The proposed methodology is to extract the cepstral features from different types of Indian audio voice and train the model of speaker identification through Indian voices. The dataset available is in US and UK English accent by the Americans, Europeans and Chinese. But Indian audio dataset is not present in rich amount. Somehow the audio features that is used to determine the different speakers vary in terms of accent of people from country to country. Hence we aim to build the maximum audio dataset for the Indian voices to help the system sustain in the Indian market and produce better results.
We aim to generate the result to predict the gender of the speaker along with the identity of the speaker. 
With good results for single speaker analysis, we plan to move forward to multiple speaker analysis and identify different speakers when more than one speaker is there in front of mic and communicating with each other. 
To pre-emphasis speech signal, a high pass filter is implemented in this process. To achieve stationary signal, audio is divided into segments (frames) with fixed duration, each with of 480 samples. 
Silence is removed by testing the energy value of each frame respect to certain threshold value to determine which frame is not silent. Hamming windowing process is applied on overlapped frames.
Sampling is done and then audio clip is divided into segments to further process for clustering process. The small chunks are analysed to match the feature vectors and form clusters
Then for each segment analysis is made where its segment of a new speaker or the segment of the previous segment speaker.

### Research Paper
https://www.ijeat.org/wp-content/uploads/papers/v8i3/C5964028319.pdf
