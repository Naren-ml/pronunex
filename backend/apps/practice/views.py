"""
API Views for practice sessions and assessments.
"""

import logging
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from django.utils import timezone

from .models import UserSession, Attempt, PhonemeError
from .serializers import (
    UserSessionSerializer,
    AttemptListSerializer,
    AttemptDetailSerializer,
    AttemptCreateSerializer,
    AssessmentResultSerializer,
)
from .services import AssessmentService
from apps.library.models import ReferenceSentence, Phoneme

logger = logging.getLogger(__name__)


class UserSessionListView(generics.ListCreateAPIView):
    """
    GET /api/v1/sessions/
    POST /api/v1/sessions/
    
    List user's practice sessions or create a new one.
    """
    
    serializer_class = UserSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserSession.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class UserSessionDetailView(generics.RetrieveAPIView):
    """
    GET /api/v1/sessions/{id}/
    
    Get session details with all attempts.
    """
    
    serializer_class = UserSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserSession.objects.filter(user=self.request.user)


class EndSessionView(APIView):
    """
    POST /api/v1/sessions/{id}/end/
    
    End a practice session and calculate overall metrics.
    """
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        try:
            session = UserSession.objects.get(pk=pk, user=request.user)
        except UserSession.DoesNotExist:
            return Response(
                {'error': 'Session not found.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if session.ended_at:
            return Response(
                {'error': 'Session already ended.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        session.ended_at = timezone.now()
        session.calculate_overall_score()
        
        return Response({
            'message': 'Session ended successfully.',
            'session': UserSessionSerializer(session).data
        })


class AttemptListView(generics.ListAPIView):
    """
    GET /api/v1/attempts/
    
    List user's practice attempts.
    """
    
    serializer_class = AttemptListSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Attempt.objects.filter(
            session__user=self.request.user
        ).select_related('sentence')


class AttemptDetailView(generics.RetrieveAPIView):
    """
    GET /api/v1/attempts/{id}/
    
    Get attempt details with phoneme errors.
    """
    
    serializer_class = AttemptDetailSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Attempt.objects.filter(
            session__user=self.request.user
        ).prefetch_related('phoneme_errors')


class AssessmentView(APIView):
    """
    POST /api/v1/assess/
    
    Core pronunciation assessment endpoint.
    
    Processing flow:
    1. Receive audio + sentence_id
    2. Save audio to storage
    3. Clean audio (normalize, trim, resample)
    4. Fetch precomputed phoneme sequence from DB
    5. Run forced alignment on user audio
    6. Slice audio into phoneme segments
    7. Generate embeddings for each slice
    8. Fetch precomputed reference embeddings (cached)
    9. Calculate cosine similarity per phoneme
    10. Identify weak phonemes (score < configurable threshold)
    11. Generate LLM feedback (text only, not scoring)
    12. Save attempt and errors to DB
    13. Return structured response
    """
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        serializer = AttemptCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        sentence_id = serializer.validated_data['sentence_id']
        audio_file = serializer.validated_data['audio']
        
        try:
            sentence = ReferenceSentence.objects.get(id=sentence_id)
        except ReferenceSentence.DoesNotExist:
            return Response(
                {'error': 'Sentence not found.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get or create active session
        session = self._get_or_create_session(request.user)
        
        # Ensure reference audio exists (auto-generate if missing)
        self._ensure_reference_audio(sentence)
        
        # Run assessment pipeline
        assessment_service = AssessmentService()
        result = assessment_service.process_attempt(audio_file, sentence)
        
        # Handle expected error cases (return 200 so frontend can display them)
        if not result.get('success', False):
            error_type = result.get('error')
            
            # Content mismatch - user said wrong sentence
            if error_type == 'content_mismatch':
                logger.info(f"Content mismatch for user {request.user.email}: {result.get('transcribed')}")
                return Response({
                    'success': False,
                    'error': 'content_mismatch',
                    'message': result.get('message', 'Speech did not match expected sentence.'),
                    'transcribed': result.get('transcribed', ''),
                    'expected': sentence.text,
                    'similarity': result.get('similarity', 0.0),
                    'suggestion': result.get('suggestion', 'Please try again.'),
                }, status=status.HTTP_200_OK)
            
            # Unscorable - technical failure
            elif error_type == 'unscorable':
                logger.warning(f"Unscorable attempt for user {request.user.email}: {result.get('reason')}")
                return Response({
                    'success': False,
                    'error': 'unscorable',
                    'message': result.get('message', 'Could not analyze audio.'),
                    'reason': result.get('reason', 'unknown'),
                    'transcribed': result.get('transcribed', ''),
                    'suggestion': result.get('suggestion', 'Please try again.'),
                }, status=status.HTTP_200_OK)
            
            # Unknown error - return 500
            logger.error(f"Assessment failed for user {request.user.email}: {result}")
            return Response(
                {'error': error_type or 'Assessment failed.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Save attempt to database
        attempt = self._save_attempt(session, sentence, audio_file, result)
        
        # Save phoneme errors
        self._save_phoneme_errors(attempt, result.get('phoneme_scores', []))
        
        # Update analytics (UserProgress, PhonemeProgress, StreakRecord) in real-time
        from apps.analytics.services import AnalyticsService
        analytics_service = AnalyticsService()
        analytics_service.update_after_attempt(request.user, attempt)
        
        logger.info(f"Assessment completed for user {request.user.email}: score {result['overall_score']}")
        
        response_data = {
            'success': True,
            'overall_score': result['overall_score'],
            'fluency_score': result.get('fluency_score'),
            'phoneme_scores': result['phoneme_scores'],
            'weak_phonemes': result['weak_phonemes'],
            'llm_feedback': result['llm_feedback'],
            'mistakes': result.get('mistakes'),
            'transcribed': result.get('transcribed'),
            'processing_time_ms': result['processing_time_ms'],
            'attempt_id': attempt.id,
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    def _get_or_create_session(self, user):
        """Get active session or create new one."""
        # Find active session (no end time, created today)
        today = timezone.now().date()
        session = UserSession.objects.filter(
            user=user,
            ended_at__isnull=True,
            started_at__date=today
        ).first()
        
        if not session:
            session = UserSession.objects.create(
                user=user,
                session_type='practice'
            )
        
        return session
    
    def _save_attempt(self, session, sentence, audio_file, result):
        """Save attempt record to database."""
        attempt = Attempt.objects.create(
            session=session,
            sentence=sentence,
            audio_file=audio_file,
            score=result['overall_score'],
            fluency_score=result.get('fluency_score'),
            phoneme_scores=result.get('phoneme_scores'),
            llm_feedback=result.get('llm_feedback'),
            processing_time_ms=result.get('processing_time_ms'),
        )
        return attempt
    
    def _save_phoneme_errors(self, attempt, phoneme_scores):
        """Save phoneme-level results to database."""
        from django.conf import settings
        threshold = settings.SCORING_CONFIG.get('WEAK_PHONEME_THRESHOLD', 0.7)
        
        for ps in phoneme_scores:
            try:
                phoneme = Phoneme.objects.get(arpabet=ps.get('phoneme'))
                PhonemeError.objects.create(
                    attempt=attempt,
                    target_phoneme=phoneme,
                    similarity_score=ps.get('score', 0),
                    word_context=ps.get('word', ''),
                    position_in_word=ps.get('position', 'medial'),
                    start_time=ps.get('start'),
                    end_time=ps.get('end'),
                )
            except Phoneme.DoesNotExist:
                logger.warning(f"Phoneme not found: {ps.get('phoneme')}")
    
    def _ensure_reference_audio(self, sentence):
        """Ensure reference audio exists, generate via TTS if missing."""
        import os
        
        audio_path = sentence.get_audio_source()
        
        # Check if audio file exists
        if audio_path and os.path.exists(audio_path):
            return  # Audio already exists
        
        # Generate TTS audio
        try:
            from services.tts_service import generate_sentence_audio
            logger.info(f"Auto-generating TTS for sentence {sentence.id}")
            audio_path = generate_sentence_audio(sentence)
            
            # Update sentence with new audio path
            from django.conf import settings
            relative_path = os.path.relpath(audio_path, settings.MEDIA_ROOT)
            sentence.audio_file = relative_path
            sentence.save(update_fields=['audio_file'])
            
            logger.info(f"TTS generated successfully for sentence {sentence.id}")
        except Exception as e:
            logger.error(f"TTS generation failed for sentence {sentence.id}: {str(e)}")
            # Continue anyway - will fall back to dev mode in assessment

