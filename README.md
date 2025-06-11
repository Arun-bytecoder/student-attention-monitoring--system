# 🎓 Student Attention Monitoring System 🧠📊

An AI-powered classroom assistant that helps teachers understand and improve student engagement **in real-time**, while respecting **student privacy** and promoting a more **inclusive learning environment**.

---

## 🚀 Overview

In modern classrooms, maintaining student attention is a growing challenge. Our project introduces a **non-intrusive AI-based monitoring system** that detects real-time engagement using standard webcams — no internet, no cloud, and no facial recognition.

By analyzing **eye movements, head pose, and facial expressions**, this tool provides teachers with a simple, color-coded dashboard to track student focus — enhancing the learning experience without disrupting the class.

---

## 🎯 Objectives

- 🧑‍🏫 **Assist teachers** in identifying disengaged students instantly.
- 🔍 **Monitor attention levels** through gaze tracking, posture, and expressions.
- 🛡️ **Preserve privacy** with fully local processing (no data leaves the classroom).
- 📈 **Improve learning outcomes** through adaptive real-time feedback.

---

## 🛠️ Technologies Used

| Module                          | Technology / Library        |
|-------------------------------|-----------------------------|
| Face Detection & Recognition  | Dlib, OpenCV                |
| Facial Landmark Detection     | Dlib (68 landmarks model)   |
| Gaze & Expression Analysis    | FER-2013, Eye Aspect Ratio  |
| Attention Scoring             | Custom Weighted Algorithm   |
| UI and Visualization          | OpenCV, Python Dashboard    |
| Data Logging & Reporting      | CSV, Local Encrypted Storage|

---

## 🧠 System Architecture


---

## 📊 Key Features

- 🔵 **Color-coded Dashboard** (Green: attentive, Yellow: partial, Red: distracted)
- 👁️ **Gaze & Blink Detection** (using EAR & pupil tracking)
- 📹 **Head Pose Analysis** (for posture-based engagement)
- 🧮 **Real-time Scoring Algorithm** (combining gaze, expression, pose)
- 🔐 **Privacy-Focused** (runs completely offline on local devices)
- 💡 **Plug-and-Play** (no extra hardware needed)

---

## 📈 Performance

- ✅ **92% Accuracy** in Face Detection
- ✅ **87% Accuracy** in Engagement Classification
- ✅ Supports **30+ FPS** real-time video processing
- ✅ Up to **15 students tracked** simultaneously
- ✅ Runs on low-cost devices like **Raspberry Pi 4**
- ✅ **35% faster** teacher interventions reported
- ✅ **82% student comfort** with non-intrusive monitoring

---

## 🌱 Future Enhancements

- 🔦 Infrared Camera Support for low-light environments
- 📲 Mobile Alerts for Teachers (smartwatch/app integration)
- 🧘‍♂️ Posture-based Activity Recognition (asana analysis)
- 📚 Integration with LMS for long-term performance tracking
- 🔁 Adaptive Thresholds for activity-specific engagement

---

## 💡 Ethical Considerations

This project is built on the principles of:
- **Respect** for student autonomy
- **Data privacy** (no cloud or face recognition)
- **Transparency** and **teacher empowerment**
- **Inclusivity** across learning styles

---

## 🧑‍💻 Team

- **G. Arunachalam** - Data Science & Vision Systems  
- **S.M. Jyothilingam** - UI & Visualization  
- **M. Somanath** - Backend & Data Processing  
- **Guided by:** Prof. C. Kamatchi, Prathyusha Engineering College  

---

## 📚 References

- DLIB, OpenCV, FER-2013 Dataset  
- [IEEE & IRJET Articles on AI Engagement Monitoring]  
- [Kazemi & Sullivan, "One millisecond face alignment"]  
- [Viola-Jones Object Detection Algorithm]  
- [Learning OpenCV by Bradski and Kaehler]  


---

## 📥 How to Run

```bash
# Clone this repo
git clone https://github.com/yourusername/student-attention-monitoring.git

# Install dependencies
pip install -r requirements.txt

# Run the main application
python main.py

 License
This project is built for educational research under the MIT License.

🙌 Acknowledgments
We thank our mentor and our institution for providing the platform to explore innovative uses of AI in education.
