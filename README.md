# Technology Facilitated Transfer of Emotions and Expressions

This project is a combination of AI (Artifical Intelligence) and VR (Virtual Reality). A senario is set which is about in a virtual scence with multi-user VR, every user communicates with each other, but the user is unwilling to expose personal privacy, so the facial expression of a person is recognized in real time and passed to others through emoji. However, when someone is telling a story, the facial expression could be easily changed due to the pronunciation of words or the emotions of the user, etc. Therefore, in this project, an effective expression and transmission of facial information between people in the virtual scene is realized.

## Project Structure

A simple chat room which supports video and audio with servre-client structure is implemented.

```bash
├── README.md
├── emojis
│   ├── angry.png
│   ├── disgust.png
│   ├── happy.png
│   ├── neutral.png
│   ├── sad.png
│   ├── scared.png
│   └── surprised.png
├── haarcascade_files
│   ├── haarcascade_eye.xml
│   └── haarcascade_frontalface_default.xml
├── models
│   ├── _mini_XCEPTION.102-0.66.hdf5
│   └── cnn.py
├── main.py             --- run the chatroom with audio and video
├── chatroom
│   ├── audio_chat.py   --- audio support
│   └── video_chat.py   --- video support
└── real_time_video.py  --- test the emotion classifier
```

## Install

### Install by pip

```pip
pip3 install -r requirements.txt
```

### Install by conda

```conda
conda env create -f environment.yml
```
