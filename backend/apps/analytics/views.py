"""
API Views for analytics and progress tracking.

Views are thin HTTP handlers - business logic is in services layer.
"""

import logging
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import UserProgress
from .serializers import (
    UserProgressSerializer,
    PhonemeProgressSerializer,
    StreakSerializer,
)
from .services import AnalyticsService, AggregationService

logger = logging.getLogger(__name__)


class ProgressDashboardView(APIView):
    """
    GET /api/v1/analytics/progress/
    
    Get comprehensive progress dashboard data.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        days = int(request.query_params.get('days', 30))
        
        # Use aggregation service for all calculations
        service = AggregationService(request.user)
        stats = service.get_dashboard_stats(days=days)
        
        response_data = {
            'total_sessions': stats['session_stats'].get('total_sessions', 0),
            'total_attempts': stats['attempt_stats'].get('total_attempts', 0),
            'total_practice_minutes': stats['total_practice_minutes'],
            'overall_average_score': round(stats['attempt_stats'].get('avg_score', 0) or 0, 2),
            'current_weak_phonemes': stats['weak_phonemes'],
            'current_strong_phonemes': stats['strong_phonemes'],
            'score_trend': stats['score_trend'],
            'recent_progress': UserProgressSerializer(stats['recent_progress'], many=True).data,
            'streak': StreakSerializer(stats['streak']).data,

            'phoneme_progress': PhonemeProgressSerializer(stats['phoneme_progress'], many=True).data,
            'weekly_scores': stats['weekly_scores'],
            'weekly_labels': stats['weekly_labels'],
            'daily_goal_progress': stats.get('daily_goal_progress', 0),
        }
        
        return Response(response_data)


class PhonemeAnalyticsView(APIView):
    """
    GET /api/v1/analytics/phonemes/
    
    Get detailed per-phoneme analytics.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        service = AggregationService(request.user)
        return Response(service.get_phoneme_analytics())


class HistoryView(generics.ListAPIView):
    """
    GET /api/v1/analytics/history/
    
    Get daily progress history.
    """
    
    serializer_class = UserProgressSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        from datetime import timedelta
        from django.utils import timezone
        
        days = int(self.request.query_params.get('days', 30))
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        return UserProgress.objects.filter(
            user=self.request.user,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('-date')


class WeakPhonemesView(APIView):
    """
    GET /api/v1/analytics/weak-phonemes/
    
    Get list of user's current weak phonemes.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from django.conf import settings
        threshold = settings.SCORING_CONFIG.get('WEAK_PHONEME_THRESHOLD', 0.7)
        
        service = AggregationService(request.user)
        weak_phonemes = service.get_weak_phonemes()
        
        return Response({
            'weak_phonemes': weak_phonemes,
            'threshold': threshold,
            'count': len(weak_phonemes),
        })


# Convenience function for backward compatibility
def update_user_analytics(user, attempt):
    """
    Update analytics after a new attempt.
    Called from assessment service.
    
    This is a wrapper for AnalyticsService.update_after_attempt()
    maintained for backward compatibility.
    """
    service = AnalyticsService()
    service.update_after_attempt(user, attempt)
