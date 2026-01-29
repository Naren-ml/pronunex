"""
Aggregation Service for Pronunex Analytics.

Handles data aggregation and calculation logic for dashboard views.
"""

import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Avg, Count, Max, Sum
from django.conf import settings

from apps.analytics.models import UserProgress, PhonemeProgress, StreakRecord
from apps.practice.models import Attempt, UserSession, PhonemeError

logger = logging.getLogger(__name__)


class AggregationService:
    """
    Service class for analytics aggregation and calculations.
    
    Responsibilities:
    - Calculate dashboard statistics
    - Determine score trends
    - Identify weak/strong phonemes
    """
    
    def __init__(self, user):
        """
        Initialize with user context.
        
        Args:
            user: User instance
        """
        self.user = user
        self.threshold = settings.SCORING_CONFIG.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    def get_dashboard_stats(self, days=30):
        """
        Get comprehensive dashboard statistics.
        
        Args:
            days: Number of days to include in historical data
            
        Returns:
            dict: Dashboard data including sessions, attempts, phoneme progress
        """
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Session statistics
        session_stats = UserSession.objects.filter(
            user=self.user
        ).aggregate(
            total_sessions=Count('id'),
            total_minutes=Sum('overall_score')
        )
        
        # Attempt statistics
        attempt_stats = Attempt.objects.filter(
            session__user=self.user
        ).aggregate(
            total_attempts=Count('id'),
            avg_score=Avg('score'),
            best_score=Max('score'),
        )
        
        # Recent daily progress
        recent_progress = UserProgress.objects.filter(
            user=self.user,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('-date')[:days]
        
        # Phoneme progress
        phoneme_progress = PhonemeProgress.objects.filter(
            user=self.user
        ).select_related('phoneme').order_by('-current_score')
        
        # Identify weak and strong phonemes
        weak_phonemes = [
            pp.phoneme.arpabet 
            for pp in phoneme_progress 
            if pp.current_score < self.threshold
        ]
        
        strong_phonemes = [
            pp.phoneme.arpabet 
            for pp in phoneme_progress 
            if pp.current_score >= 0.85
        ]
        
        # Calculate score trend
        score_trend = self._calculate_score_trend(recent_progress)
        
        # Get or create streak
        streak, _ = StreakRecord.objects.get_or_create(user=self.user)
        
        # Calculate weekly scores (last 7 days)
        # We need strict last 7 days filling gaps with 0
        weekly_scores = []
        weekly_labels = []
        
        # Create a lookup dict for recent progress
        progress_map = {p.date: p for p in recent_progress}
        
        for i in range(6, -1, -1):
            day = end_date - timedelta(days=i)
            day_progress = progress_map.get(day)
            
            score = 0
            if day_progress and day_progress.average_score:
                # Convert 0.0-1.0 to 0-100 range for visualization
                score = round(day_progress.average_score * 100, 1)
            
            weekly_scores.append(score)
            weekly_labels.append(day.strftime("%a"))

        # Calculate total practice minutes
        total_minutes = sum(p.total_practice_minutes for p in recent_progress)
        
        return {
            'session_stats': session_stats,
            'attempt_stats': attempt_stats,
            'recent_progress': recent_progress,
            'phoneme_progress': phoneme_progress,
            'weak_phonemes': weak_phonemes[:10],
            'strong_phonemes': strong_phonemes[:10],
            'score_trend': score_trend,
            'streak': streak,
            'weekly_scores': weekly_scores,
            'weekly_labels': weekly_labels,
            'total_practice_minutes': round(total_minutes, 1),
        }
    
    def _calculate_score_trend(self, recent_progress):
        """
        Calculate score trend based on recent progress.
        
        Args:
            recent_progress: QuerySet of UserProgress records
            
        Returns:
            str: 'improving', 'declining', 'stable', or 'insufficient_data'
        """
        if len(recent_progress) < 3:
            return 'insufficient_data'
        
        recent_scores = [
            rp.average_score for rp in recent_progress[:3] 
            if rp.average_score
        ]
        older_scores = [
            rp.average_score for rp in recent_progress[3:6] 
            if rp.average_score
        ]
        
        if not recent_scores or not older_scores:
            return 'insufficient_data'
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg + 0.05:
            return 'improving'
        elif recent_avg < older_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def get_weak_phonemes(self):
        """
        Get list of user's current weak phonemes.
        
        Returns:
            list: Weak phoneme details with scores
        """
        phoneme_progress = PhonemeProgress.objects.filter(
            user=self.user,
            current_score__lt=self.threshold
        ).select_related('phoneme').order_by('current_score')
        
        return [
            {
                'phoneme': pp.phoneme.arpabet,
                'symbol': pp.phoneme.symbol,
                'score': round(pp.current_score, 2),
                'attempts': pp.attempts_count,
            }
            for pp in phoneme_progress
        ]
    
    def get_phoneme_analytics(self):
        """
        Get detailed per-phoneme analytics grouped by type.
        
        Returns:
            dict: Phoneme analytics by type with weak phoneme details
        """
        phoneme_progress = PhonemeProgress.objects.filter(
            user=self.user
        ).select_related('phoneme').order_by('phoneme__type', 'phoneme__arpabet')
        
        # Group by phoneme type
        by_type = {}
        for pp in phoneme_progress:
            ptype = pp.phoneme.type
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append({
                'phoneme': pp.phoneme.arpabet,
                'symbol': pp.phoneme.symbol,
                'current_score': pp.current_score,
                'attempts': pp.attempts_count,
                'best_score': pp.best_score,
            })
        
        # Get weak phoneme details with error contexts
        weak_details = []
        for pp in phoneme_progress.filter(current_score__lt=self.threshold):
            errors = PhonemeError.objects.filter(
                attempt__session__user=self.user,
                target_phoneme=pp.phoneme
            ).values('word_context', 'position_in_word').annotate(
                count=Count('id'),
                avg_score=Avg('similarity_score')
            ).order_by('-count')[:5]
            
            weak_details.append({
                'phoneme': pp.phoneme.arpabet,
                'symbol': pp.phoneme.symbol,
                'current_score': pp.current_score,
                'attempts': pp.attempts_count,
                'common_errors': list(errors),
            })
        
        return {
            'by_type': by_type,
            'weak_phonemes_detail': weak_details,
            'total_phonemes_practiced': phoneme_progress.count(),
        }
