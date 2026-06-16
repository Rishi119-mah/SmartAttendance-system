import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class AttendanceML:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.iso_forest = IsolationForest(contamination=0.2, random_state=42)
        self.is_rf_trained = False
        
    def _calculate_streak(self, dates):
        # Calculate max consecutive absent days in last 30 days
        if not dates:
            return 0
        dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in dates])
        if not dates:
            return 0
            
        today = datetime.now()
        streak = 0
        max_streak = 0
        
        # Check last 30 days
        for i in range(30):
            d = today - timedelta(days=i)
            # Skip weekends (5=Sat, 6=Sun)
            if d.weekday() > 4:
                continue
            
            d_str = d.strftime("%Y-%m-%d")
            if d_str not in [x.strftime("%Y-%m-%d") for x in dates]:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _get_student_features(self, student_name, df):
        # df has columns: student_id, student_name, date, time
        student_df = df[df['student_name'] == student_name]
        
        if df.empty:
            total_days = 1
        else:
            total_days = max(1, df['date'].nunique())
            
        present_days = student_df['date'].nunique()
        attendance_rate = present_days / total_days
        
        # Calculate streak
        streak_absent = self._calculate_streak(student_df['date'].tolist())
        
        # Trend slope (last 10 days)
        recent_dates = sorted(df['date'].unique())[-10:]
        if len(recent_dates) >= 2:
            y = [1 if d in student_df['date'].values else 0 for d in recent_dates]
            X = np.arange(len(y)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, y)
            trend_slope = lr.coef_[0]
        else:
            trend_slope = 0
            
        # Day of week patterns
        student_df = student_df.copy()
        if not student_df.empty:
            student_df['date_obj'] = pd.to_datetime(student_df['date'])
            student_df['weekday'] = student_df['date_obj'].dt.weekday
            dow_counts = student_df['weekday'].value_counts()
        else:
            dow_counts = pd.Series(dtype=int)
            
        # Total possible days per weekday in the dataset
        if not df.empty:
            df_temp = df.copy()
            df_temp['date_obj'] = pd.to_datetime(df_temp['date'])
            total_dow = df_temp[['date', 'date_obj']].drop_duplicates()['date_obj'].dt.weekday.value_counts()
        else:
            total_dow = pd.Series(dtype=int)
            
        dow_rates = []
        for i in range(5): # Mon-Fri
            pres = dow_counts.get(i, 0)
            tot = total_dow.get(i, 1)
            if tot == 0: tot = 1
            dow_rates.append(pres / tot)
            
        features = {
            'attendance_rate': attendance_rate,
            'streak_absent': streak_absent,
            'trend_slope': trend_slope,
            'monday_rate': dow_rates[0],
            'tuesday_rate': dow_rates[1],
            'wednesday_rate': dow_rates[2],
            'thursday_rate': dow_rates[3],
            'friday_rate': dow_rates[4]
        }
        return features

    def _generate_synthetic_data(self):
        # Generate 200 synthetic records
        X_syn = []
        y_syn = []
        
        for _ in range(200):
            # Base rate
            rate = np.random.uniform(0.3, 1.0)
            
            # Streak inversely correlated with rate
            streak = int(max(0, (1.0 - rate) * 15 + np.random.normal(0, 2)))
            
            # Slope
            slope = np.random.uniform(-0.1, 0.1)
            if rate < 0.5: slope = np.random.uniform(-0.2, 0)
            if rate > 0.9: slope = np.random.uniform(0, 0.2)
                
            # DOW rates
            dow = [max(0, min(1, rate + np.random.uniform(-0.2, 0.2))) for _ in range(5)]
            
            # Add strict noise
            rate = max(0, min(1, rate + np.random.uniform(-0.05, 0.05)))
            
            # Labels: <0.60 -> 2 (HIGH), 0.60-0.75 -> 1 (MEDIUM), >0.75 -> 0 (LOW)
            if rate < 0.60:
                label = 2
            elif rate <= 0.75:
                label = 1
            else:
                label = 0
                
            features = [rate, streak, slope] + dow
            X_syn.append(features)
            y_syn.append(label)
            
        return np.array(X_syn), np.array(y_syn)

    def predict_risk(self, student_name, attendance_df):
        if not self.is_rf_trained:
            X_train, y_train = self._generate_synthetic_data()
            self.rf_model.fit(X_train, y_train)
            self.is_rf_trained = True
            
        total_system_days = attendance_df['date'].nunique() if not attendance_df.empty else 0
        
        if total_system_days < 3:
            feat_dict = self._get_student_features(student_name, attendance_df)
            return {
                "risk_level": "LOW",
                "risk_confidence": 100.0,
                "feature_importances": {"attendance_rate": 0, "streak_absent": 0, "trend_slope": 0, "dow_pattern": 0},
                "attendance_rate": feat_dict['attendance_rate'],
                "streak_absent": feat_dict['streak_absent']
            }
            
        feat_dict = self._get_student_features(student_name, attendance_df)
        X_test = np.array([[
            feat_dict['attendance_rate'],
            feat_dict['streak_absent'],
            feat_dict['trend_slope'],
            feat_dict['monday_rate'],
            feat_dict['tuesday_rate'],
            feat_dict['wednesday_rate'],
            feat_dict['thursday_rate'],
            feat_dict['friday_rate']
        ]])
        
        pred = self.rf_model.predict(X_test)[0]
        proba = self.rf_model.predict_proba(X_test)[0]
        
        levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        risk_level = levels[pred]
        risk_confidence = float(max(proba) * 100)
        
        importances = self.rf_model.feature_importances_
        feature_importances = {
            "attendance_rate": importances[0],
            "streak_absent": importances[1],
            "trend_slope": importances[2],
            "dow_pattern": sum(importances[3:])
        }
        
        return {
            "risk_level": risk_level,
            "risk_confidence": risk_confidence,
            "feature_importances": feature_importances,
            "attendance_rate": feat_dict['attendance_rate'],
            "streak_absent": feat_dict['streak_absent']
        }

    def project_trend(self, history_df):
        if history_df.empty:
            return {"slope": 0, "projection": [], "trend_direction": "stable"}
            
        # Group by date
        daily_counts = history_df.groupby('date').size().reset_index(name='count')
        daily_counts['date_obj'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.sort_values('date_obj')
        
        # Calculate rolling 7-day attendance count
        # For simplicity since dates might skip, we'll just use a rolling window on the available days
        daily_counts['rolling_7'] = daily_counts['count'].rolling(window=7, min_periods=1).mean()
        
        if len(daily_counts) < 2:
            return {"slope": 0, "projection": [], "trend_direction": "stable"}
            
        X = np.arange(len(daily_counts)).reshape(-1, 1)
        y = daily_counts['rolling_7'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        slope = lr.coef_[0]
        
        # Project 7 days forward
        last_date = daily_counts['date_obj'].iloc[-1]
        last_x = X[-1][0]
        
        projection = []
        for i in range(1, 8):
            proj_date = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
            proj_val = max(0, lr.predict([[last_x + i]])[0])
            projection.append({"date": proj_date, "value": float(proj_val)})
            
        if slope > 0.5:
            trend_dir = "improving"
        elif slope < -0.5:
            trend_dir = "declining"
        else:
            trend_dir = "stable"
            
        return {
            "slope": float(slope),
            "projection": projection,
            "trend_direction": trend_dir
        }

    def detect_anomalies(self, all_students_df):
        if all_students_df.empty:
            return []
            
        students = all_students_df['student_name'].unique()
        if len(students) < 5:
            return [] # Need a few students for isolation forest to make sense
            
        X = []
        student_list = []
        
        for st in students:
            feat = self._get_student_features(st, all_students_df)
            
            # calculate weekly variance roughly
            st_df = all_students_df[all_students_df['student_name'] == st]
            if not st_df.empty:
                st_df = st_df.copy()
                st_df['date_obj'] = pd.to_datetime(st_df['date'])
                st_df['week'] = st_df['date_obj'].dt.isocalendar().week
                weekly_var = st_df.groupby('week').size().var()
                if pd.isna(weekly_var): weekly_var = 0
            else:
                weekly_var = 0
                
            X.append([feat['attendance_rate'], feat['streak_absent'], feat['trend_slope'], weekly_var])
            student_list.append(st)
            
        X = np.array(X)
        self.iso_forest.fit(X)
        preds = self.iso_forest.predict(X)
        scores = self.iso_forest.decision_function(X)
        
        results = []
        for i in range(len(preds)):
            if preds[i] == -1: # Anomaly
                rate = X[i][0]
                streak = X[i][1]
                slope = X[i][2]
                var = X[i][3]
                
                reason = "General irregular pattern"
                if streak > 5:
                    reason = f"Sudden {int(streak)}-day absence streak detected"
                elif slope < -0.2:
                    reason = "Sharp drop in attendance trend recently"
                elif var > 2:
                    reason = "Highly irregular weekly attendance pattern"
                    
                results.append({
                    "student_name": student_list[i],
                    "anomaly_score": float(scores[i]),
                    "reason": reason,
                    "is_anomaly": True
                })
        return results

    def day_of_week_pattern(self, attendance_df):
        if attendance_df.empty:
            return {}
            
        df = attendance_df.copy()
        df['date_obj'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date_obj'].dt.weekday
        
        days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
        
        # We only care about Mon-Fri
        df = df[df['weekday'] < 5]
        
        # Calculate rates
        total_unique_dates = df[['date', 'weekday']].drop_duplicates()
        date_counts = total_unique_dates['weekday'].value_counts()
        
        present_counts = df['weekday'].value_counts()
        
        patterns = {}
        for i in range(5):
            total_days_for_dow = date_counts.get(i, 0)
            if total_days_for_dow == 0:
                patterns[days_map[i]] = 0.0
            else:
                # Approximate rate: total presents / (total students * total_days_for_dow)
                # But since we just want relative rates, let's do present_counts / total_days_for_dow
                total_students = df['student_id'].nunique()
                if total_students == 0: total_students = 1
                rate = present_counts.get(i, 0) / (total_students * total_days_for_dow)
                patterns[days_map[i]] = round(float(rate), 2)
                
        if patterns:
            best_day = max(patterns, key=patterns.get)
            worst_day = min(patterns, key=patterns.get)
            patterns['best_day'] = best_day
            patterns['worst_day'] = worst_day
            
        return patterns

    def class_heatmap_data(self, attendance_df):
        if attendance_df.empty:
            return {"students": [], "dates": [], "matrix": []}
            
        # Pivot: students as rows, dates as columns, value = 1/0
        df = attendance_df.copy()
        df['present'] = 1
        
        pivot = df.pivot_table(index='student_name', columns='date', values='present', fill_value=0)
        
        dates = pivot.columns.tolist()
        students = pivot.index.tolist()
        matrix = pivot.values.tolist()
        
        return {
            "students": students,
            "dates": dates,
            "matrix": matrix
        }
