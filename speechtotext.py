def speech_recog():
    # [START speech_quickstart]
    import io
    import os

    # Imports the Google Cloud client library
    from google.cloud import speech

    # Instantiates a client
    speech_client = speech.Client()
    
    # Flag to check detection
    detected = True

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        'RecordedFile.wav')

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        sample = speech_client.sample(
            content,
            source_uri=None,
            encoding='LINEAR16')

    # Detects speech in the audio file
    try:
        alternatives = sample.recognize('ko-KR')
        # alternatives = sample.recognize('en-US')
    except ValueError:
        detected = False

    outFile = open('result/result.txt', 'w')
    if detected:
        for alternative in alternatives:
            outFile.write(alternative.transcript)
    else:
        outFile.write('No Voice Detected.')

    outFile.close()

    # [END speech_quickstart]


if __name__ == '__main__':
    speech_recog()