import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AttendanceNLG:
    def __init__(self):
        # --- Multi-template pools ---
        
        self.SUMMARY_TEMPLATES = {
            "good": [
                "Overall attendance is strong this week at {rate}%. The majority of students ({today_count}/{total_students}) are consistently present.",
                "We are seeing excellent attendance figures. The current class average sits at a healthy {rate}%, with {today_count} out of {total_students} students present today.",
                "The class is demonstrating great commitment, achieving an overall attendance rate of {rate}%. Today's turnout was {today_count} of {total_students} students."
            ],
            "average": [
                "Attendance is moderate this week at {rate}%. Today, {today_count} out of {total_students} students were present.",
                "We have an average attendance rate of {rate}% currently. {today_count} out of {total_students} students attended today's session.",
                "The attendance metrics show a stable but average rate of {rate}%. We saw {today_count} of {total_students} students today."
            ],
            "poor": [
                "Attendance has dropped to a concerning {rate}%. Only {today_count} out of {total_students} students were present today.",
                "We are experiencing low attendance, currently averaging {rate}%. Today's turnout was just {today_count} of {total_students} students.",
                "Urgent attention is needed as overall attendance has fallen to {rate}%. Only {today_count} of {total_students} students attended today."
            ]
        }

        self.RISK_TEMPLATES = {
            "mild": [
                "Students showing mild risk (60-75%) include: {students}. Early engagement is recommended.",
                "The following students are dipping into the mild risk category (60-75%): {students}.",
                "Monitor these students closely as their attendance is between 60-75%: {students}."
            ],
            "severe": [
                "CRITICAL: The following students have severe attendance issues (<60%): {students}.",
                "Immediate intervention required for students with <60% attendance: {students}.",
                "These students are failing to meet basic attendance requirements (<60%): {students}."
            ],
            "critical": [
                "URGENT: {students} have missed 3 or more consecutive days.",
                "The following students are on a critical absence streak (3+ days): {students}.",
                "Alert: {students} have been absent for at least 3 consecutive days."
            ]
        }

        self.HIGHLIGHT_TEMPLATES = [
            "Outstanding attendance noted for: {students}. Their dedication is commendable.",
            "Special recognition to {students} for maintaining perfect or near-perfect attendance.",
            "The following students are leading the class in attendance: {students}. Great job!"
        ]

        self.ACTION_TEMPLATES = [
            "Recommendation: Schedule 1-on-1 meetings with severe risk students. Send automated parent notes for those on critical streaks.",
            "Action Plan: Follow up immediately with students absent for 3+ days. Reward top attenders with positive reinforcement.",
            "Suggested Intervention: Issue warning letters to all severe risk students and monitor mild risk students for further decline."
        ]

        self.LETTER_TEMPLATES = {
            "opening": [
                "Dear {name},",
                "To {name},",
                "Attention {name},"
            ],
            "body": [
                "We are writing to inform you that your current attendance rate has dropped to {rate}%. You have missed a total of {missed_total} days, including {missed_days}.",
                "This letter serves as an official notice regarding your attendance. Your current rate is {rate}%. You have been absent on {missed_days} (Total: {missed_total} days).",
                "Our records indicate a concerning attendance pattern. Your attendance rate sits at {rate}%, having missed {missed_total} days: {missed_days}."
            ],
            "closing": [
                "Please meet with your advisor immediately to discuss this matter.",
                "Consistent attendance is vital for your success. We urge you to improve your attendance immediately.",
                "Failure to improve may result in further academic consequences. Please reach out if you require support."
            ]
        }

        self.PARENT_NOTE_TEMPLATES = [
            "Dear Parent/Guardian,\nWe wanted to share an update regarding {name}'s attendance. Their current attendance rate is {rate}%, having missed {missed_total} days. Regular attendance is crucial for academic success. Please discuss this with {name} and reach out if we can assist in any way.",
            "Dear Parent/Guardian,\nThis is a courtesy notice that {name} has an attendance rate of {rate}%. They have been absent for {missed_total} days recently. We want to ensure {name} stays on track. Please contact us to discuss any challenges they might be facing.",
            "Dear Parent/Guardian,\nWe are reaching out because {name}'s attendance has fallen to {rate}%. Missing {missed_total} days can impact their learning progress. We appreciate your partnership in encouraging {name} to attend regularly."
        ]

    def generate_report(self, stats: dict) -> str:
        """
        stats keys: total_students, today_count, overall_rate, at_risk_list,
        top_students, trend_direction, day_counts (dict of weekday→count)
        """
        rate = stats.get('overall_rate', 0)
        
        # Summary Section
        if rate > 85:
            summary = random.choice(self.SUMMARY_TEMPLATES["good"])
        elif rate >= 70:
            summary = random.choice(self.SUMMARY_TEMPLATES["average"])
        else:
            summary = random.choice(self.SUMMARY_TEMPLATES["poor"])
            
        summary = summary.format(rate=rate, today_count=stats.get('today_count', 0), total_students=stats.get('total_students', 0))

        # Risks Section
        risk_str = ""
        at_risk_list = stats.get('at_risk_list', [])
        if at_risk_list:
            mild = [s['name'] for s in at_risk_list if s.get('risk_level') == 'mild']
            severe = [s['name'] for s in at_risk_list if s.get('risk_level') == 'severe']
            critical = [s['name'] for s in at_risk_list if s.get('risk_level') == 'critical']
            
            if critical:
                risk_str += random.choice(self.RISK_TEMPLATES["critical"]).format(students=", ".join(critical)) + " "
            if severe:
                risk_str += random.choice(self.RISK_TEMPLATES["severe"]).format(students=", ".join(severe)) + " "
            if mild:
                risk_str += random.choice(self.RISK_TEMPLATES["mild"]).format(students=", ".join(mild))
        else:
            risk_str = "No students are currently flagged as at-risk."

        # Highlights Section
        top_students = stats.get('top_students', [])
        if top_students:
            highlights = random.choice(self.HIGHLIGHT_TEMPLATES).format(students=", ".join(top_students))
        else:
            highlights = "No outstanding highlights for this period."

        # Trend Section
        trend_direction = stats.get('trend_direction', 'stable')
        day_counts = stats.get('day_counts', {})
        trend = f"The recent attendance trend is currently showing a {trend_direction} trajectory. "
        if day_counts:
            highest_day = max(day_counts, key=day_counts.get)
            lowest_day = min(day_counts, key=day_counts.get)
            trend += f"Peak attendance usually occurs on {highest_day}, while {lowest_day} sees the most absences."
        else:
            trend += "Insufficient daily data to determine weekday patterns."

        # Actions Section
        actions = random.choice(self.ACTION_TEMPLATES)

        report = f"""**Attendance Summary**
{summary}

**Risk Assessment**
{risk_str}

**Class Highlights**
{highlights}

**Trend Analysis**
{trend}

**Recommended Actions**
{actions}"""

        return report

    def generate_warning_letter(self, student_stats: dict) -> str:
        """
        student_stats keys: name, attendance_rate, missed_days (list),
        streak_absent, total_days
        """
        name = student_stats.get('name', 'Student')
        rate = student_stats.get('attendance_rate', 0)
        missed_days = student_stats.get('missed_days', [])
        missed_total = len(missed_days)
        streak = student_stats.get('streak_absent', 0)
        
        missed_str = ", ".join(missed_days[:5])
        if len(missed_days) > 5:
            missed_str += f" and {len(missed_days)-5} other days"

        opening = random.choice(self.LETTER_TEMPLATES["opening"]).format(name=name)
        body = random.choice(self.LETTER_TEMPLATES["body"]).format(rate=rate, missed_total=missed_total, missed_days=missed_str)
        
        # Add streak warning if critical
        if streak >= 3:
            body += f" We specifically noted that you have missed {streak} consecutive days."
            
        closing = random.choice(self.LETTER_TEMPLATES["closing"])
        
        letter = f"""{opening}

{body}

{closing}

Sincerely,
Administration"""
        return letter

    def generate_parent_note(self, student_stats: dict) -> str:
        """
        Shorter, warmer tone than warning letter.
        """
        name = student_stats.get('name', 'Student')
        rate = student_stats.get('attendance_rate', 0)
        missed_total = len(student_stats.get('missed_days', []))
        
        note = random.choice(self.PARENT_NOTE_TEMPLATES).format(name=name, rate=rate, missed_total=missed_total)
        return note

    def find_similar_week(self, current_stats: dict, db_summaries: list) -> str:
        """
        - Convert current_stats to descriptive sentence string
        - TF-IDF vectorize (TfidfVectorizer) all stored summary_text entries
        - Compute cosine_similarity between current description and all past summaries
        - Return the week_start of the most similar past week
        - Handle edge case: <2 past summaries → return "No comparable week found"
        """
        if not db_summaries or len(db_summaries) < 2:
            return "No comparable week found"
            
        rate = current_stats.get('overall_rate', 0)
        trend = current_stats.get('trend_direction', 'stable')
        current_desc = f"Attendance rate is {rate} percent with a {trend} trend."
        
        texts = [s.get('summary_text', '') for s in db_summaries]
        week_starts = [s.get('week_start', '') for s in db_summaries]
        
        # Append current description at the end
        texts.append(current_desc)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute cosine similarity of the last item against all others
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            # Find index of highest similarity
            best_match_idx = cosine_sim.argmax()
            
            # Ensure similarity is somewhat meaningful (>0.1)
            if cosine_sim[best_match_idx] > 0.1:
                return week_starts[best_match_idx]
            else:
                return "No comparable week found"
        except Exception:
            return "No comparable week found"
