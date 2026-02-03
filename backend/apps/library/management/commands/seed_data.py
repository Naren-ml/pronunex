"""
Management command to seed phoneme data.

Loads the 44 English phonemes into the database.
Run with: python manage.py seed_data
"""

from django.core.management.base import BaseCommand
from apps.library.models import Phoneme, ReferenceSentence


class Command(BaseCommand):
    help = 'Seed database with English phonemes and initial data'
    
    def handle(self, *args, **options):
        self.stdout.write('Seeding phoneme data...')
        
        phonemes_created = self.seed_phonemes()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {phonemes_created} phonemes')
        )
        
        self.stdout.write('Seeding reference sentences...')
        sentences_created = self.seed_sentences()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {sentences_created} sentences')
        )
    
    def seed_phonemes(self):
        """Seed all 44 English phonemes."""
        
        phonemes_data = [
            # Vowels
            {'symbol': '/i/', 'arpabet': 'IY', 'ipa': 'i', 'type': 'vowel', 'example_word': 'see', 'description': 'High front tense vowel'},
            {'symbol': '/ɪ/', 'arpabet': 'IH', 'ipa': 'ɪ', 'type': 'vowel', 'example_word': 'sit', 'description': 'High front lax vowel'},
            {'symbol': '/e/', 'arpabet': 'EY', 'ipa': 'eɪ', 'type': 'diphthong', 'example_word': 'say', 'description': 'Mid front diphthong'},
            {'symbol': '/ɛ/', 'arpabet': 'EH', 'ipa': 'ɛ', 'type': 'vowel', 'example_word': 'bed', 'description': 'Mid front lax vowel'},
            {'symbol': '/æ/', 'arpabet': 'AE', 'ipa': 'æ', 'type': 'vowel', 'example_word': 'cat', 'description': 'Low front vowel'},
            {'symbol': '/ɑ/', 'arpabet': 'AA', 'ipa': 'ɑ', 'type': 'vowel', 'example_word': 'father', 'description': 'Low back vowel'},
            {'symbol': '/ɔ/', 'arpabet': 'AO', 'ipa': 'ɔ', 'type': 'vowel', 'example_word': 'thought', 'description': 'Mid back rounded vowel'},
            {'symbol': '/o/', 'arpabet': 'OW', 'ipa': 'oʊ', 'type': 'diphthong', 'example_word': 'go', 'description': 'Mid back diphthong'},
            {'symbol': '/ʊ/', 'arpabet': 'UH', 'ipa': 'ʊ', 'type': 'vowel', 'example_word': 'book', 'description': 'High back lax vowel'},
            {'symbol': '/u/', 'arpabet': 'UW', 'ipa': 'u', 'type': 'vowel', 'example_word': 'too', 'description': 'High back tense vowel'},
            {'symbol': '/ʌ/', 'arpabet': 'AH', 'ipa': 'ʌ', 'type': 'vowel', 'example_word': 'but', 'description': 'Mid central vowel'},
            {'symbol': '/ə/', 'arpabet': 'AX', 'ipa': 'ə', 'type': 'vowel', 'example_word': 'about', 'description': 'Schwa (reduced vowel)'},
            {'symbol': '/ɝ/', 'arpabet': 'ER', 'ipa': 'ɝ', 'type': 'vowel', 'example_word': 'bird', 'description': 'R-colored vowel'},
            
            # Diphthongs
            {'symbol': '/aɪ/', 'arpabet': 'AY', 'ipa': 'aɪ', 'type': 'diphthong', 'example_word': 'my', 'description': 'Low front to high diphthong'},
            {'symbol': '/aʊ/', 'arpabet': 'AW', 'ipa': 'aʊ', 'type': 'diphthong', 'example_word': 'how', 'description': 'Low to high back diphthong'},
            {'symbol': '/ɔɪ/', 'arpabet': 'OY', 'ipa': 'ɔɪ', 'type': 'diphthong', 'example_word': 'boy', 'description': 'Mid back to high front diphthong'},
            
            # Plosives (Stops)
            {'symbol': '/p/', 'arpabet': 'P', 'ipa': 'p', 'type': 'plosive', 'example_word': 'pen', 'description': 'Voiceless bilabial plosive'},
            {'symbol': '/b/', 'arpabet': 'B', 'ipa': 'b', 'type': 'plosive', 'example_word': 'bad', 'description': 'Voiced bilabial plosive'},
            {'symbol': '/t/', 'arpabet': 'T', 'ipa': 't', 'type': 'plosive', 'example_word': 'ten', 'description': 'Voiceless alveolar plosive'},
            {'symbol': '/d/', 'arpabet': 'D', 'ipa': 'd', 'type': 'plosive', 'example_word': 'day', 'description': 'Voiced alveolar plosive'},
            {'symbol': '/k/', 'arpabet': 'K', 'ipa': 'k', 'type': 'plosive', 'example_word': 'key', 'description': 'Voiceless velar plosive'},
            {'symbol': '/g/', 'arpabet': 'G', 'ipa': 'g', 'type': 'plosive', 'example_word': 'go', 'description': 'Voiced velar plosive'},
            
            # Fricatives
            {'symbol': '/f/', 'arpabet': 'F', 'ipa': 'f', 'type': 'fricative', 'example_word': 'fan', 'description': 'Voiceless labiodental fricative'},
            {'symbol': '/v/', 'arpabet': 'V', 'ipa': 'v', 'type': 'fricative', 'example_word': 'van', 'description': 'Voiced labiodental fricative'},
            {'symbol': '/θ/', 'arpabet': 'TH', 'ipa': 'θ', 'type': 'fricative', 'example_word': 'think', 'description': 'Voiceless dental fricative'},
            {'symbol': '/ð/', 'arpabet': 'DH', 'ipa': 'ð', 'type': 'fricative', 'example_word': 'this', 'description': 'Voiced dental fricative'},
            {'symbol': '/s/', 'arpabet': 'S', 'ipa': 's', 'type': 'fricative', 'example_word': 'see', 'description': 'Voiceless alveolar fricative'},
            {'symbol': '/z/', 'arpabet': 'Z', 'ipa': 'z', 'type': 'fricative', 'example_word': 'zoo', 'description': 'Voiced alveolar fricative'},
            {'symbol': '/ʃ/', 'arpabet': 'SH', 'ipa': 'ʃ', 'type': 'fricative', 'example_word': 'she', 'description': 'Voiceless postalveolar fricative'},
            {'symbol': '/ʒ/', 'arpabet': 'ZH', 'ipa': 'ʒ', 'type': 'fricative', 'example_word': 'measure', 'description': 'Voiced postalveolar fricative'},
            {'symbol': '/h/', 'arpabet': 'HH', 'ipa': 'h', 'type': 'fricative', 'example_word': 'hat', 'description': 'Voiceless glottal fricative'},
            
            # Affricates
            {'symbol': '/tʃ/', 'arpabet': 'CH', 'ipa': 'tʃ', 'type': 'affricate', 'example_word': 'chin', 'description': 'Voiceless postalveolar affricate'},
            {'symbol': '/dʒ/', 'arpabet': 'JH', 'ipa': 'dʒ', 'type': 'affricate', 'example_word': 'jam', 'description': 'Voiced postalveolar affricate'},
            
            # Nasals
            {'symbol': '/m/', 'arpabet': 'M', 'ipa': 'm', 'type': 'nasal', 'example_word': 'man', 'description': 'Bilabial nasal'},
            {'symbol': '/n/', 'arpabet': 'N', 'ipa': 'n', 'type': 'nasal', 'example_word': 'no', 'description': 'Alveolar nasal'},
            {'symbol': '/ŋ/', 'arpabet': 'NG', 'ipa': 'ŋ', 'type': 'nasal', 'example_word': 'sing', 'description': 'Velar nasal'},
            
            # Liquids
            {'symbol': '/l/', 'arpabet': 'L', 'ipa': 'l', 'type': 'liquid', 'example_word': 'leg', 'description': 'Alveolar lateral approximant'},
            {'symbol': '/r/', 'arpabet': 'R', 'ipa': 'ɹ', 'type': 'liquid', 'example_word': 'red', 'description': 'Alveolar approximant'},
            
            # Glides (Semivowels)
            {'symbol': '/w/', 'arpabet': 'W', 'ipa': 'w', 'type': 'glide', 'example_word': 'we', 'description': 'Labial-velar approximant'},
            {'symbol': '/j/', 'arpabet': 'Y', 'ipa': 'j', 'type': 'glide', 'example_word': 'yes', 'description': 'Palatal approximant'},
        ]
        
        # Add articulation tips
        tips = {
            'TH': 'Place tongue tip between teeth, blow air gently without voice.',
            'DH': 'Same as TH but add voice from your throat.',
            'R': 'Curl tongue tip back slightly, do not touch roof of mouth.',
            'L': 'Touch tongue tip to ridge behind upper front teeth.',
            'SH': 'Round lips slightly, push air through wide channel.',
            'CH': 'Start with tongue at roof, release with SH sound.',
            'S': 'Tongue behind teeth, narrow air channel, no voice.',
            'Z': 'Same as S but with voice.',
            'NG': 'Back of tongue touches soft palate, air through nose.',
            'W': 'Round lips tightly, glide to next vowel.',
            'V': 'Gently bite lower lip, blow air with voice.',
            'F': 'Same as V but without voice.',
        }
        
        created_count = 0
        for data in phonemes_data:
            data['articulation_tip'] = tips.get(data['arpabet'], '')
            phoneme, created = Phoneme.objects.update_or_create(
                arpabet=data['arpabet'],
                defaults=data
            )
            if created:
                created_count += 1
        
        return created_count
    
    def seed_sentences(self):
        """Seed reference sentences for practice."""
        
        sentences_data = [
            # Beginner sentences (10 alliterative sentences)
            {
                'text': 'Big blue balls bounce brightly.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['B', 'IH', 'G', 'B', 'L', 'UW', 'B', 'AO', 'L', 'Z', 'B', 'AW', 'N', 'S', 'B', 'R', 'AY', 'T', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['B', 'L', 'R'],
                'source': 'curated',
            },
            {
                'text': 'Tiny turtles tiptoe together.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['T', 'AY', 'N', 'IY', 'T', 'ER', 'T', 'AH', 'L', 'Z', 'T', 'IH', 'P', 'T', 'OW', 'T', 'AH', 'G', 'EH', 'DH', 'ER'],
                'alignment_map': [],
                'target_phonemes': ['T', 'P', 'G'],
                'source': 'curated',
            },
            {
                'text': 'Red rabbits run rapidly.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['R', 'EH', 'D', 'R', 'AE', 'B', 'IH', 'T', 'S', 'R', 'AH', 'N', 'R', 'AE', 'P', 'IH', 'D', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['R', 'B', 'P'],
                'source': 'curated',
            },
            {
                'text': 'Silly snakes slide slowly.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['S', 'IH', 'L', 'IY', 'S', 'N', 'EY', 'K', 'S', 'S', 'L', 'AY', 'D', 'S', 'L', 'OW', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['S', 'L', 'N'],
                'source': 'curated',
            },
            {
                'text': 'Funny frogs flip fast.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['F', 'AH', 'N', 'IY', 'F', 'R', 'AA', 'G', 'Z', 'F', 'L', 'IH', 'P', 'F', 'AE', 'S', 'T'],
                'alignment_map': [],
                'target_phonemes': ['F', 'R', 'L'],
                'source': 'curated',
            },
            {
                'text': 'Busy bees buzz loudly.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['B', 'IH', 'Z', 'IY', 'B', 'IY', 'Z', 'B', 'AH', 'Z', 'L', 'AW', 'D', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['B', 'Z', 'L'],
                'source': 'curated',
            },
            {
                'text': 'Happy hens hop home.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['HH', 'AE', 'P', 'IY', 'HH', 'EH', 'N', 'Z', 'HH', 'AA', 'P', 'HH', 'OW', 'M'],
                'alignment_map': [],
                'target_phonemes': ['HH', 'P', 'N'],
                'source': 'curated',
            },
            {
                'text': 'Cute cats clap quietly.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['K', 'Y', 'UW', 'T', 'K', 'AE', 'T', 'S', 'K', 'L', 'AE', 'P', 'K', 'W', 'AY', 'AH', 'T', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['K', 'L', 'W'],
                'source': 'curated',
            },
            {
                'text': 'Little lions lick lemons.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['L', 'IH', 'T', 'AH', 'L', 'L', 'AY', 'AH', 'N', 'Z', 'L', 'IH', 'K', 'L', 'EH', 'M', 'AH', 'N', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['L', 'K', 'M'],
                'source': 'curated',
            },
            {
                'text': 'Green grapes grow great.',
                'difficulty_level': 'beginner',
                'phoneme_sequence': ['G', 'R', 'IY', 'N', 'G', 'R', 'EY', 'P', 'S', 'G', 'R', 'OW', 'G', 'R', 'EY', 'T'],
                'alignment_map': [],
                'target_phonemes': ['G', 'R', 'P'],
                'source': 'curated',
            },
            # Intermediate sentences (10 alliterative sentences)
            {
                'text': 'Clever clowns clap cleanly and quickly.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['K', 'L', 'EH', 'V', 'ER', 'K', 'L', 'AW', 'N', 'Z', 'K', 'L', 'AE', 'P', 'K', 'L', 'IY', 'N', 'L', 'IY', 'AE', 'N', 'D', 'K', 'W', 'IH', 'K', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['K', 'L', 'W'],
                'source': 'curated',
            },
            {
                'text': 'Seven slippery snakes slid silently south.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['S', 'EH', 'V', 'AH', 'N', 'S', 'L', 'IH', 'P', 'ER', 'IY', 'S', 'N', 'EY', 'K', 'S', 'S', 'L', 'IH', 'D', 'S', 'AY', 'L', 'AH', 'N', 'T', 'L', 'IY', 'S', 'AW', 'TH'],
                'alignment_map': [],
                'target_phonemes': ['S', 'L', 'TH'],
                'source': 'curated',
            },
            {
                'text': 'Brave brown bears baked bread briskly.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['B', 'R', 'EY', 'V', 'B', 'R', 'AW', 'N', 'B', 'EH', 'R', 'Z', 'B', 'EY', 'K', 'T', 'B', 'R', 'EH', 'D', 'B', 'R', 'IH', 'S', 'K', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['B', 'R', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Noisy nurses need neat notes nightly.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['N', 'OY', 'Z', 'IY', 'N', 'ER', 'S', 'AH', 'Z', 'N', 'IY', 'D', 'N', 'IY', 'T', 'N', 'OW', 'T', 'S', 'N', 'AY', 'T', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['N', 'Z', 'T'],
                'source': 'curated',
            },
            {
                'text': 'Three thin thieves threw thick threads.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['TH', 'R', 'IY', 'TH', 'IH', 'N', 'TH', 'IY', 'V', 'Z', 'TH', 'R', 'UW', 'TH', 'IH', 'K', 'TH', 'R', 'EH', 'D', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['TH', 'R', 'V'],
                'source': 'curated',
            },
            {
                'text': 'Friendly foxes fixed fresh fences.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['F', 'R', 'EH', 'N', 'D', 'L', 'IY', 'F', 'AA', 'K', 'S', 'AH', 'Z', 'F', 'IH', 'K', 'S', 'T', 'F', 'R', 'EH', 'SH', 'F', 'EH', 'N', 'S', 'AH', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['F', 'R', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Six sticky sweets stuck suddenly.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['S', 'IH', 'K', 'S', 'S', 'T', 'IH', 'K', 'IY', 'S', 'W', 'IY', 'T', 'S', 'S', 'T', 'AH', 'K', 'S', 'AH', 'D', 'AH', 'N', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['S', 'T', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Lazy lizards love long lunches.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['L', 'EY', 'Z', 'IY', 'L', 'IH', 'Z', 'ER', 'D', 'Z', 'L', 'AH', 'V', 'L', 'AO', 'NG', 'L', 'AH', 'N', 'CH', 'AH', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['L', 'Z', 'NG'],
                'source': 'curated',
            },
            {
                'text': 'Proud parents prepare perfect picnics.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['P', 'R', 'AW', 'D', 'P', 'EH', 'R', 'AH', 'N', 'T', 'S', 'P', 'R', 'IH', 'P', 'EH', 'R', 'P', 'ER', 'F', 'IH', 'K', 'T', 'P', 'IH', 'K', 'N', 'IH', 'K', 'S'],
                'alignment_map': [],
                'target_phonemes': ['P', 'R', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Bright butterflies beat blue blossoms.',
                'difficulty_level': 'intermediate',
                'phoneme_sequence': ['B', 'R', 'AY', 'T', 'B', 'AH', 'T', 'ER', 'F', 'L', 'AY', 'Z', 'B', 'IY', 'T', 'B', 'L', 'UW', 'B', 'L', 'AA', 'S', 'AH', 'M', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['B', 'R', 'L'],
                'source': 'curated',
            },
            # Advanced sentences (10 alliterative sentences)
            {
                'text': 'Six sleek swans swiftly swam southward silently.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['S', 'IH', 'K', 'S', 'S', 'L', 'IY', 'K', 'S', 'W', 'AA', 'N', 'Z', 'S', 'W', 'IH', 'F', 'T', 'L', 'IY', 'S', 'W', 'AE', 'M', 'S', 'AW', 'TH', 'W', 'ER', 'D', 'S', 'AY', 'L', 'AH', 'N', 'T', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['S', 'W', 'TH'],
                'source': 'curated',
            },
            {
                'text': 'Crisp crackers crack crazily under crushing crunches.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['K', 'R', 'IH', 'S', 'P', 'K', 'R', 'AE', 'K', 'ER', 'Z', 'K', 'R', 'AE', 'K', 'K', 'R', 'EY', 'Z', 'AH', 'L', 'IY', 'AH', 'N', 'D', 'ER', 'K', 'R', 'AH', 'SH', 'IH', 'NG', 'K', 'R', 'AH', 'N', 'CH', 'AH', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['K', 'R', 'SH'],
                'source': 'curated',
            },
            {
                'text': 'Thirty trembling teachers taught tricky tongue twisters.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['TH', 'ER', 'T', 'IY', 'T', 'R', 'EH', 'M', 'B', 'AH', 'L', 'IH', 'NG', 'T', 'IY', 'CH', 'ER', 'Z', 'T', 'AO', 'T', 'T', 'R', 'IH', 'K', 'IY', 'T', 'AH', 'NG', 'T', 'W', 'IH', 'S', 'T', 'ER', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['TH', 'T', 'R'],
                'source': 'curated',
            },
            {
                'text': "Fred's fresh fried fish fries frightfully fast.",
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['F', 'R', 'EH', 'D', 'Z', 'F', 'R', 'EH', 'SH', 'F', 'R', 'AY', 'D', 'F', 'IH', 'SH', 'F', 'R', 'AY', 'Z', 'F', 'R', 'AY', 'T', 'F', 'AH', 'L', 'IY', 'F', 'AE', 'S', 'T'],
                'alignment_map': [],
                'target_phonemes': ['F', 'R', 'SH'],
                'source': 'curated',
            },
            {
                'text': 'Strict strong students stress smart study strategies.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['S', 'T', 'R', 'IH', 'K', 'T', 'S', 'T', 'R', 'AO', 'NG', 'S', 'T', 'UW', 'D', 'AH', 'N', 'T', 'S', 'S', 'T', 'R', 'EH', 'S', 'S', 'M', 'AA', 'R', 'T', 'S', 'T', 'AH', 'D', 'IY', 'S', 'T', 'R', 'AE', 'T', 'AH', 'JH', 'IY', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['S', 'T', 'R'],
                'source': 'curated',
            },
            {
                'text': 'Rare red roses really reek rapidly.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['R', 'EH', 'R', 'R', 'EH', 'D', 'R', 'OW', 'Z', 'AH', 'Z', 'R', 'IY', 'L', 'IY', 'R', 'IY', 'K', 'R', 'AE', 'P', 'AH', 'D', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['R', 'Z', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Shy shells shimmer sharply in shallow shores.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['SH', 'AY', 'SH', 'EH', 'L', 'Z', 'SH', 'IH', 'M', 'ER', 'SH', 'AA', 'R', 'P', 'L', 'IY', 'IH', 'N', 'SH', 'AE', 'L', 'OW', 'SH', 'AO', 'R', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['SH', 'L', 'R'],
                'source': 'curated',
            },
            {
                'text': 'Black blocks bounce back before breaking badly.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['B', 'L', 'AE', 'K', 'B', 'L', 'AA', 'K', 'S', 'B', 'AW', 'N', 'S', 'B', 'AE', 'K', 'B', 'IH', 'F', 'AO', 'R', 'B', 'R', 'EY', 'K', 'IH', 'NG', 'B', 'AE', 'D', 'L', 'IY'],
                'alignment_map': [],
                'target_phonemes': ['B', 'L', 'K'],
                'source': 'curated',
            },
            {
                'text': 'Furious fast frogs frequently flip from fences.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['F', 'Y', 'UH', 'R', 'IY', 'AH', 'S', 'F', 'AE', 'S', 'T', 'F', 'R', 'AA', 'G', 'Z', 'F', 'R', 'IY', 'K', 'W', 'AH', 'N', 'T', 'L', 'IY', 'F', 'L', 'IH', 'P', 'F', 'R', 'AH', 'M', 'F', 'EH', 'N', 'S', 'AH', 'Z'],
                'alignment_map': [],
                'target_phonemes': ['F', 'R', 'L'],
                'source': 'curated',
            },
            {
                'text': 'Wicked whispering winds whir wildly westward.',
                'difficulty_level': 'advanced',
                'phoneme_sequence': ['W', 'IH', 'K', 'AH', 'D', 'W', 'IH', 'S', 'P', 'ER', 'IH', 'NG', 'W', 'IH', 'N', 'D', 'Z', 'W', 'ER', 'W', 'AY', 'L', 'D', 'L', 'IY', 'W', 'EH', 'S', 'T', 'W', 'ER', 'D'],
                'alignment_map': [],
                'target_phonemes': ['W', 'S', 'NG'],
                'source': 'curated',
            },
        ]
        
        created_count = 0
        for data in sentences_data:
            sentence, created = ReferenceSentence.objects.update_or_create(
                text=data['text'],
                defaults={
                    'difficulty_level': data['difficulty_level'],
                    'phoneme_sequence': data['phoneme_sequence'],
                    'alignment_map': data['alignment_map'],
                    'target_phonemes': data['target_phonemes'],
                    'source': data['source'],
                    'is_validated': True,
                }
            )
            if created:
                created_count += 1
        
        return created_count
