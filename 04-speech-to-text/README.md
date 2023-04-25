# Speech to text

https://platform.openai.com/docs/guides/speech-to-text

* In this lab we will practice how to turn audio into text

### Introduction:

* The speech to text API provides two endpoints, transcriptions and translations, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:

    - Transcribe audio into whatever language the audio is in.

    - Translate and transcribe the audio into english.

* File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm.

## Quickstart

### Transcriptions

* The transcriptions API takes as input the audio file you want to transcribe and the desired output file format for the transcription of the audio

### Translations

* The translations API takes as input the audio file in any of the supported languages and transcribes, if necessary, the audio into english.

* This differs from our /Transcriptions endpoint since the output is not in the original input language and is instead translated to english text.

1. Python installed on your machine
2. Valid OpenAI-API key 

### Step 1) Bla bla

# Speech to text

https://platform.openai.com/docs/guides/speech-to-text

* In this lab we will practice how to turn audio into text

### Introduction:

* The speech to text API provides two endpoints, transcriptions and translations, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:

    - Transcribe audio into whatever language the audio is in.

    - Translate and transcribe the audio into english.

* File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm.

## Quickstart

### Transcriptions

* The transcriptions API takes as input the audio file you want to transcribe and the desired output file format for the transcription of the audio

* Download the mp3 files from git location

    - <git location for mp3 file>

* open terminal and enter into python console using below command

    - python

* Copy below code in python terminal

``` python
    import openai
    openai.api_key = "<your open api key>"
    audio_file= open("/home/ubuntu/mp3_neverendingstory-german.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
```

* you will see the out put like this

``` json
    {
    "text": "Es war einmal ein Mann, der hatte sieben S\u00f6hne. Die sieben S\u00f6hne sprachen, Vater, erz\u00e4hl uns eine Geschichte. Da fing der Vater an. Es war einmal ein Mann, der hatte sieben S\u00f6hne. Die sieben S\u00f6hne sprachen, Vater, erz\u00e4hl uns eine Geschichte. Da fing der Vater an. Es war einmal ein Mann, der hatte sieben S\u00f6hne. die sieben S\u00f6hne sprachen."
    }
```

### Translations

* The translations API takes as input the audio file in any of the supported languages and transcribes, if necessary, the audio into english.

* This differs from our /Transcriptions endpoint since the output is not in the original input language and is instead translated to english text.

* Download the mp3 files from git location

    - <git location for mp3 file>

* open terminal and enter into python console using below command

    - python

* Copy below code in python terminal

``` python
    import openai
    openai.api_key = "<your open api key>"
    audio_file= open("/home/ubuntu/mp3_neverendingstory-german.mp3", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    print(transcript)
```

* you will see the out put like this

``` json
    {
    "text": "once upon a time, there was a man who had seven sons. Father! Tell us a story. The father started... Once upon a time, there was a man who had seven sons. The seven sons would say... Father! Tell us a story. Then the father said... Once upon a time, there was a man who had seven sons. The seven sons spoke."
    }
```

* OpenAI only support translation into English at this time

## Longer inputs
* By default, the Whisper API only supports files that are less than 25 MB.

* If you have an audio file that is longer than that, you will need to break it up into chunks of 25 MB's or less or used a compressed audio format.

* To get the best performance, we suggest that you avoid breaking the audio up mid-sentence as this may cause some context to be lost.

MP3 files are available at this Github location "https://github.com/elephantscale/terraform-up-and-running-code/tree/master/mp3"

# Speech to text

https://platform.openai.com/docs/guides/speech-to-text

* In this lab we will practice how to turn audio into text

### Introduction:

* The speech to text API provides two endpoints, transcriptions and translations, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:

    - Transcribe audio into whatever language the audio is in.

    - Translate and transcribe the audio into english.

* File uploads are currently limited to 25 MB and the following input file types are supported: mp3, mp4, mpeg, mpga, m4a, wav, and webm.

## Quickstart

### Transcriptions

* The transcriptions API takes as input the audio file you want to transcribe and the desired output file format for the transcription of the audio

* Download the mp3 files from git location

    - <git location for mp3 file>

* open terminal and install open ai library using below command:

    ```
        pip install openai
    ```

* Enter into python console using below command

    - python

* Copy below code in python terminal

``` python
    import openai
    openai.api_key = "<your open api key>"
    audio_file= open("/home/ubuntu/mp3_neverendingstory-german.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
```

* you will see the out put like this

``` json
    {
    "text": "Es war einmal ein Mann, der hatte sieben S\u00f6hne. Die sieben S\u00f6hne sprachen, Vater, erz\u00e4hl uns eine Geschichte. Da fing der Vater an. Es war einmal ein Mann, der hatte sieben S\u00f6hne. Die sieben S\u00f6hne sprachen, Vater, erz\u00e4hl uns eine Geschichte. Da fing der Vater an. Es war einmal ein Mann, der hatte sieben S\u00f6hne. die sieben S\u00f6hne sprachen."
    }
```

### Translations

* The translations API takes as input the audio file in any of the supported languages and transcribes, if necessary, the audio into english.

* This differs from our /Transcriptions endpoint since the output is not in the original input language and is instead translated to english text.

* Download the mp3 files from git location

    - <git location for mp3 file>

* open terminal and enter into python console using below command

    - python

* Copy below code in python terminal

``` python
    import openai
    openai.api_key = "<your open api key>"
    audio_file= open("/home/ubuntu/mp3_neverendingstory-german.mp3", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    print(transcript)
```

* you will see the out put like this

``` json
    {
    "text": "once upon a time, there was a man who had seven sons. Father! Tell us a story. The father started... Once upon a time, there was a man who had seven sons. The seven sons would say... Father! Tell us a story. Then the father said... Once upon a time, there was a man who had seven sons. The seven sons spoke."
    }
```

* OpenAI only support translation into English at this time

## Longer inputs
* By default, the Whisper API only supports files that are less than 25 MB.

* If you have an audio file that is longer than that, you will need to break it up into chunks of 25 MB's or less or used a compressed audio format.

* To get the best performance, we suggest that you avoid breaking the audio up mid-sentence as this may cause some context to be lost.

