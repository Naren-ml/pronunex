"""
Analytics Service for Pronunex.

Handles user analytics updates after pronunciation attempts.
Extracted from views.py for better separation of concerns.
"""

import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Avg, Max

from apps.analytics.models import UserProgress, PhonemeProgress, StreakRecord
from apps.practice.models import Attempt, PhonemeError

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Service class for updating user analytics.
    
    Responsibilities:
    - Update daily progress after attempts
    - Track phoneme-level progress
    - Manage practice streaks
    """
    
    def update_after_attempt(self, user, attempt):
        """
        Update all analytics after a new attempt.
        
        Args:
            user: User instance
            attempt: Attempt instance with phoneme_errors
        """
        self._update_daily_progress(user)
        self._update_phoneme_progress(user, attempt)
        self._update_streak(user)
        
        logger.debug(f"Analytics updated for user after attempt")
    
    def _update_daily_progress(self, user):
        """Update daily aggregated progress."""
        today = timezone.now().date()
        
        progress, created = UserProgress.objects.get_or_create(
            user=user,
            date=today,
            defaults={
                'sessions_count': 0,
                'attempts_count': 0,
                'average_score': 0,
            }
        )
        
        # Increment attempt count
        progress.attempts_count += 1
        
        # Recalculate today's averages
        today_stats = Attempt.objects.filter(
            session__user=user,
            created_at__date=today
        ).aggregate(
            avg=Avg('score'),
            best=Max('score')
        )
        
        progress.average_score = today_stats.get('avg', 0)
        progress.best_score = today_stats.get('best', 0)
        progress.save()
    
    def _update_phoneme_progress(self, user, attempt):
        """Update per-phoneme progress records."""
        for error in attempt.phoneme_errors.all():
            phoneme_prog, _ = PhonemeProgress.objects.get_or_create(
                user=user,
                phoneme=error.target_phoneme,
                defaults={'first_practiced': timezone.now()}
            )
            
            # Update attempt count and timestamp
            phoneme_prog.attempts_count += 1
            phoneme_prog.last_practiced = timezone.now()
            
            # Track first attempt score
            if phoneme_prog.first_attempt_score is None:
                phoneme_prog.first_attempt_score = error.similarity_score
            
            # Update best score
            if phoneme_prog.best_score is None or error.similarity_score > phoneme_prog.best_score:
                phoneme_prog.best_score = error.similarity_score
            
            # Calculate current score as recent average (last 10 attempts)
            recent_errors = PhonemeError.objects.filter(
                attempt__session__user=user,
                target_phoneme=error.target_phoneme
            ).order_by('-attempt__created_at')[:10]
            
            if recent_errors:
                phoneme_prog.current_score = sum(
                    e.similarity_score for e in recent_errors
                ) / len(recent_errors)
            
            phoneme_prog.save()
    
    def _update_streak(self, user):
        """Update practice streak for gamification."""
        today = timezone.now().date()
        
        streak, _ = StreakRecord.objects.get_or_create(user=user)
        
        if streak.last_practice_date:
            if streak.last_practice_date == today:
                pass  # Already practiced today
            elif streak.last_practice_date == today - timedelta(days=1):
                streak.current_streak += 1
            else:
                streak.current_streak = 1  # Reset streak
        else:
            streak.current_streak = 1
        
        streak.last_practice_date = today
        
        if streak.current_streak > streak.longest_streak:
            streak.longest_streak = streak.current_streak
        
        streak.save()
