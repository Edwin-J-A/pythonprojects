This project presents a highly integrated, hands-free control system that allows users to interact with their computers using facial gestures and voice commands. By combining multiple technologies including MediaPipe Face Mesh, Dlib, and speech recognition.  The system enables seamless desktop navigation without the need for traditional input devices.


Cursor movement is achieved through nose tip tracking, providing intuitive directional control based on head position. Eye blinks, detected via Eye Aspect Ratio (EAR), are used to perform click actions, with precise thresholds ensuring accuracy across various screen elements, including taskbar and window controls. Mouth gestures, monitored using the Mouth Aspect Ratio (MAR), allow for vertical scrolling, with added logic to lock scrolling if the mouth remains open for an extended period.


A robust voice command system activates via a hotword and supports commands for zooming, window management, tab control, and program state changes. The system includes profile-based customization - Default, Presentation, Silent, and Accessibility modes, each optimizing the interface for different use cases. Text-to-speech feedback, powered by pyttsx3, enhances the user experience with real-time confirmations.


The program incorporates lighting detection to halt execution under poor conditions and runs voice recognition in a background thread to maintain responsiveness during pauses. Designed with accessibility in mind, this system empowers users with limited mobility to control their computers effectively and serves as a scalable platform for future innovations in assistive technology and intelligent human-computer interaction.
