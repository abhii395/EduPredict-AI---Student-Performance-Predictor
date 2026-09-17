"""
ai_insights.py - Mistral AI integration for personalized study recommendations
Uses codestral-2508 model via the mistralai SDK.
"""

import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', '')
MISTRAL_MODEL   = 'codestral-2508'


def generate_insights(student_data: dict, prediction: dict) -> str:
    """
    Generate personalized AI insights and recommendations for the student.

    Args:
        student_data: Raw form input (gender, race_ethnicity, parental_education,
                      lunch, test_prep)
        prediction:   Output from predict.run_prediction()

    Returns:
        A markdown-formatted string with insights and recommendations.
    """
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == 'your_mistral_api_key_here':
        return _fallback_insights(student_data, prediction)

    grade   = prediction['grade_prediction']
    letter  = prediction['letter_grade']
    passed  = prediction['pass_fail'] == 1
    pass_p  = prediction.get('pass_probability')

    prompt = f"""You are an expert educational counselor and data scientist. Analyze this student's profile and predicted performance, then provide actionable, empathetic, and specific recommendations.

## Student Profile
- **Gender**: {student_data.get('gender', 'N/A')}
- **Parental Education**: {student_data.get('parental_education', 'N/A')}
- **Study Hours per Week**: {student_data.get('study_hours', 'N/A')}
- **Attendance Rate**: {student_data.get('attendance_rate', 'N/A')}%
- **Past Exam Score**: {student_data.get('past_exam_score', 'N/A')} / 100
- **Internet Access**: {student_data.get('internet_access', 'N/A')}
- **Extracurricular Activities**: {student_data.get('extracurricular_activities', 'N/A')}

## Predicted Performance
- **Predicted Final Score**: {grade:.1f}/100
- **Letter Grade**: {letter}
- **Predicted Outcome**: {'PASS ✅' if passed else 'FAIL ❌'}
- **Pass Probability**: {f'{pass_p:.1f}%' if pass_p is not None else 'N/A'}

## Task
Provide a structured response with the following sections in markdown:

### 📊 Performance Analysis
Brief analysis of what the prediction means and what key factors (e.g. attendance, study hours, past performance) influenced it.

### 🎯 Personalized Study Recommendations
3-5 specific, actionable study tips tailored to this student's profile. Pay special attention to their reported study hours and attendance.

### 📚 Focus Areas
Based on typical patterns for students with this profile, what should they focus on?

### 💡 Motivational Insight
An empathetic, encouraging message specific to this student's situation.

### 🚀 Next Steps
2-3 concrete immediate actions the student can take this week.

Keep the response concise, warm, and actionable. Use emojis sparingly but effectively."""

    try:
        client   = Mistral(api_key=MISTRAL_API_KEY)
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a compassionate educational advisor who provides '
                        'data-driven, personalized academic recommendations. '
                        'Always be encouraging and specific.'
                    )
                },
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI insights temporarily unavailable: {str(e)}\n\n{_fallback_insights(student_data, prediction)}"


def _fallback_insights(student_data: dict, prediction: dict) -> str:
    """Generate rule-based insights when Mistral API is unavailable."""
    grade   = prediction['grade_prediction']
    letter  = prediction['letter_grade']
    passed  = prediction['pass_fail'] == 1
    
    study_hours = float(student_data.get('study_hours', 0))
    attendance = float(student_data.get('attendance_rate', 0))

    tips = []
    if attendance < 85:
        tips.append("📝 **Improve attendance** — students with >85% attendance score significantly higher. Try not to miss classes.")
    if study_hours < 15:
        tips.append("⏰ **Increase study time** — dedicating at least 15 hours a week can make a big difference in your score.")
    if student_data.get('extracurricular_activities') == 'No':
        tips.append("⚽ **Consider extracurriculars** — balanced involvement can improve time management and reduce stress.")
    if grade < 60:
        tips.append("🔄 **Seek tutoring** — personalized help in core subjects can make a significant difference.")
    elif grade < 80:
        tips.append("📖 **Practice past papers** — regular mock tests help reinforce concepts and improve timing.")
    else:
        tips.append("🏆 **Challenge yourself** — consider advanced coursework to stay engaged and excel further.")


    status_msg = "You are on track to pass! 🎉" if passed else "With focused effort, improvement is absolutely achievable. 💪"

    return f"""### 📊 Performance Analysis
Predicted average score of **{grade:.1f}/100** (Grade **{letter}**). {status_msg}

### 🎯 Personalized Study Recommendations
{chr(10).join(tips)}

### 📚 Subject Focus Areas
Based on your profile, focus on **Mathematics** and **Writing** — these subjects typically show the highest improvement with structured preparation.

### 💡 Motivational Insight
Every student has unique strengths. Your background and experiences are assets — use them to bring diverse perspectives to your studies.

### 🚀 Next Steps
1. Review your weakest subject this week for 30 minutes daily.
2. Find an accountability partner or study group.
3. Set one small, measurable goal for the next 7 days.

---
*Note: Add your Mistral API key to `.env` for AI-powered personalized insights.*"""
