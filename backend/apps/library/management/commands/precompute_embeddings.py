"""
Django management command to precompute reference embeddings for all sentences.

This significantly speeds up pronunciation assessment by caching
phoneme-level embeddings for reference audio.
"""

import logging
from django.core.management.base import BaseCommand
from django.db import transaction
from apps.library.models import ReferenceSentence
from nlp_core.vectorizer import batch_audio_to_embeddings
from nlp_core.audio_slicer import slice_audio_by_timestamps
from nlp_core.aligner import get_phoneme_timestamps_with_text
import pickle
import os

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Precompute reference embeddings for all sentences in the library'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Recompute embeddings even if they already exist',
        )
        parser.add_argument(
            '--sentence-id',
            type=int,
            help='Precompute only for specific sentence ID',
        )

    def handle(self, *args, **options):
        force = options['force']
        sentence_id = options.get('sentence_id')

        # Get sentences to process
        if sentence_id:
            sentences = ReferenceSentence.objects.filter(id=sentence_id)
            if not sentences.exists():
                self.stdout.write(self.style.ERROR(
                    f'Sentence with ID {sentence_id} not found'
                ))
                return
        else:
            sentences = ReferenceSentence.objects.all()

        total = sentences.count()
        self.stdout.write(self.style.SUCCESS(
            f'Found {total} sentence(s) to process'
        ))

        processed = 0
        skipped = 0
        failed = 0

        for sentence in sentences:
            # Skip if embeddings already exist and not forcing
            if sentence.reference_embeddings and not force:
                self.stdout.write(self.style.WARNING(
                    f'[{sentence.id}] Skipping "{sentence.text[:30]}..." '
                    f'(embeddings already cached)'
                ))
                skipped += 1
                continue

            self.stdout.write(
                f'[{sentence.id}] Processing: "{sentence.text[:50]}..."'
            )

            try:
                # Step 1: Check if reference audio exists
                if not sentence.audio_file or not os.path.exists(sentence.audio_file.path):
                    self.stdout.write(self.style.ERROR(
                        f'  ✗ No reference audio file found'
                    ))
                    failed += 1
                    continue

                audio_path = sentence.audio_file.path
                self.stdout.write(f'  → Audio: {os.path.basename(audio_path)}')

                # Step 2: Get phoneme timestamps using forced alignment
                self.stdout.write('  → Running forced alignment...')
                phoneme_timestamps = get_phoneme_timestamps_with_text(
                    audio_path,
                    sentence.text,
                    expected_phonemes=sentence.phoneme_sequence
                )

                if not phoneme_timestamps:
                    self.stdout.write(self.style.ERROR(
                        '  ✗ Failed to get phoneme timestamps'
                    ))
                    failed += 1
                    continue

                self.stdout.write(f'  → Found {len(phoneme_timestamps)} phonemes')

                # Step 3: Slice audio into phoneme segments
                self.stdout.write('  → Slicing audio...')
                audio_slices = slice_audio_by_timestamps(
                    audio_path,
                    phoneme_timestamps
                )

                if not audio_slices:
                    self.stdout.write(self.style.ERROR(
                        '  ✗ Failed to slice audio'
                    ))
                    failed += 1
                    continue

                # Step 4: Generate embeddings for each slice
                self.stdout.write('  → Generating embeddings...')
                embeddings = batch_audio_to_embeddings(audio_slices)

                if not embeddings:
                    self.stdout.write(self.style.ERROR(
                        '  ✗ Failed to generate embeddings'
                    ))
                    failed += 1
                    continue

                # Step 5: Serialize and save to database
                self.stdout.write('  → Saving to database...')
                with transaction.atomic():
                    sentence.reference_embeddings = pickle.dumps(embeddings)
                    sentence.save(update_fields=['reference_embeddings'])

                self.stdout.write(self.style.SUCCESS(
                    f'  ✓ Successfully cached {len(embeddings)} embeddings'
                ))
                processed += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'  ✗ Error: {str(e)}'
                ))
                logger.exception(f'Failed to process sentence {sentence.id}')
                failed += 1
                continue

        # Summary
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS(
            f'✓ Processed: {processed}/{total}'
        ))
        if skipped > 0:
            self.stdout.write(self.style.WARNING(
                f'⊘ Skipped (already cached): {skipped}'
            ))
        if failed > 0:
            self.stdout.write(self.style.ERROR(
                f'✗ Failed: {failed}'
            ))
        self.stdout.write('='*60)

        if processed > 0:
            self.stdout.write(self.style.SUCCESS(
                '\nEmbeddings successfully precomputed! '
                'Assessment speed should now be much faster.'
            ))
