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

* Download all mp3 files from git location

    - https://github.com/elephantscale/OpenAI-labs/tree/main/mp3


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

## Supported languages

* We currently support the following languages through both the transcriptions and translations endpoint:

    - Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

## Longer inputs
* By default, the Whisper API only supports files that are less than 25 MB. 

* If you have an audio file that is longer than that, you will need to break it up into chunks of 25 MB's or less or used a compressed audio format.

* To get the best performance, we suggest that you avoid breaking the audio up mid-sentence as this may cause some context to be lost.

* One way to handle this is to use the PyDub open source Python package to split the audio:

    - Open Terminal and install pydub using below command

    ```
     pip install pydub
    ```

    - Open Terminal and install ffmpeg using below command
    ```
    sudo apt install ffmpeg
    ```

``` python
    from pydub import AudioSegment  

    song = AudioSegment.from_mp3("good_morning.mp3")

    # PyDub handles time in milliseconds
    one_minutes = 1 * 60 * 1000

    first_1_minutes = song[:one_minutes]

    first_1_minutes.export("commercials-sampleanswers-german_1.mp3", format="mp3")

```