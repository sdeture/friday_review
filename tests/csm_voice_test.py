"""
CSM Voice Integration Testing Script for AI-Aligned Therapeutic Assistant
===============================================================================

This script tests the CSM voice integration for therapeutic applications based on
Karen Horney's psychoanalytic framework. It includes:
- Basic voice generation tests
- Context handling tests
- Audio quality evaluation
- Performance benchmarking
- Edge case testing

Requirements:
- Google Colab environment with GPU access
- Access to 'sesame/csm-1b' model
- Access to 'meta-llama/Llama-3.2-1B' model
"""

import os
import time
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from google.colab import drive

# Ensure we're in the right environment
try:
    from google.colab import drive
    IN_COLAB = True
except:
    IN_COLAB = False
    print("Not running in Colab environment - some functions may be limited")

# Mount Google Drive if in Colab
if IN_COLAB:
    drive.mount('/content/drive')
    
    # Install required packages if needed
    !pip install -q transformers torchaudio==2.4.0 torch==2.4.0 huggingface_hub==0.28.1

    # Import CSM dependencies - adjust paths as needed
    from generator import load_csm_1b, Segment
else:
    print("WARNING: Imports will fail unless CSM dependencies are in path")

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# ================ Test Configuration ================

@dataclass
class TestConfig:
    output_dir: str = "/content/drive/MyDrive/psychoanalyst_assistant/test_results"
    num_runs: int = 3  # Number of runs for each test to average results
    test_all: bool = True
    run_basic_tests: bool = True
    run_context_tests: bool = True
    run_quality_tests: bool = True
    run_performance_tests: bool = True
    run_edge_tests: bool = True
    save_audio: bool = True
    plot_results: bool = True

# ================ Therapeutic Test Data ================

# Karen Horney-inspired therapeutic statements
THERAPEUTIC_STATEMENTS = {
    "short": [
        "How does that make you feel?",
        "Tell me more about that experience.",
        "I'm here to listen.",
        "What comes to mind when you think about that?",
    ],
    "medium": [
        "It sounds like you're struggling with some anxiety about this situation. In Horney's framework, this might relate to your movement toward people.",
        "Your reaction suggests you might be experiencing some inner conflict between your idealized self and your real self.",
        "The way you describe this situation reminds me of what Horney calls 'moving against people' - a tendency to approach life as a struggle.",
        "That sense of disconnection you're describing might relate to what Horney called 'detachment' - moving away from others as a defense.",
    ],
    "long": [
        "When we examine your experience through Horney's lens, we can see how this pattern developed as a coping mechanism. You may have learned to idealize certain qualities to manage anxiety. This creates a gap between your 'real self' and 'ideal self,' which often leads to the discomfort you're describing. By recognizing this pattern, we can work toward integrating these aspects of yourself.",
        "Horney would suggest that your emotional response here represents one of the three movements: moving toward people (compliance), moving against people (aggression), or moving away from people (detachment). Based on what you've shared, it seems you're primarily using the strategy of moving toward others - seeking approval and connection, perhaps at the expense of your own needs. Let's explore how this might be affecting your current situation.",
        "The internal conflict you're describing is what Horney would call 'basic anxiety' - a deep feeling of being isolated in a potentially hostile world. This often develops in childhood and can lead to various coping strategies. In your case, it seems you've developed what she called 'the search for glory' - trying to achieve an idealized version of yourself that feels impossible to reach. This creates the persistent feeling of inadequacy you mentioned.",
    ],
    "emotional": [
        "I can hear how painful that experience was for you. It takes courage to share these vulnerable feelings.",
        "Your frustration is completely understandable given what you've been through.",
        "It makes sense that you would feel overwhelmed by these conflicting emotions.",
        "I'm struck by the strength you've shown in navigating such a difficult situation.",
    ]
}

# Context examples for testing
CONTEXT_PAIRS = [
    {
        "client": "I've been feeling really anxious lately, especially at work. I can't seem to make decisions without worrying what everyone will think.",
        "therapist": "That sounds challenging. This concern about others' opinions might relate to what Horney called 'moving toward' tendencies - seeking approval and connection with others."
    },
    {
        "client": "Sometimes I just want to prove everyone wrong. Show them I don't need their help or approval.",
        "therapist": "That desire to prove yourself reminds me of what Horney described as 'moving against' others - approaching life as a kind of struggle or competition."
    },
    {
        "client": "I often feel like I just need to withdraw from everyone. Like it's safer to be alone with my thoughts.",
        "therapist": "This tendency to withdraw might represent what Horney called 'moving away' from others - creating distance as a way to find safety and avoid conflict."
    }
]

# Edge cases for testing
EDGE_CASES = {
    "very_short": ["Yes.", "I see.", "Go on.", "And?"],
    "specialized_terms": [
        "Let's explore your neurotic trends and how they manifest in your interpersonal relationships.",
        "This reflects your intrapsychic conflicts between your ideal self, despised self, and real self.",
        "The tyranny of the should is evident in how you describe your perfectionism.",
    ],
    "emotional_transitions": [
        "I understand this is painful for you. But I also see tremendous strength in how you've handled this situation.",
        "While that anxiety feels overwhelming now, we can work through this together to find a path forward.",
        "It's natural to feel anger in this situation, and at the same time, I hear a yearning for connection beneath that anger.",
    ]
}

# ================ Test Implementation ================

class CSMVoiceTest:
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = None
        self.results = {
            "basic_tests": {},
            "context_tests": {},
            "quality_tests": {},
            "performance_tests": {},
            "edge_tests": {}
        }
        self._setup_output_directory()
        
    def _setup_output_directory(self):
        """Create output directory structure if it doesn't exist"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
            print(f"Created output directory at {self.config.output_dir}")
            
        for test_type in self.results.keys():
            test_dir = os.path.join(self.config.output_dir, test_type)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
    
    def initialize_model(self):
        """Initialize the CSM model"""
        print(f"Initializing CSM model on {self.device}...")
        start_time = time.time()
        
        try:
            self.generator = load_csm_1b(device=self.device)
            init_time = time.time() - start_time
            print(f"Model initialized in {init_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            return False
    
    def get_prompt_segments(self) -> List[Segment]:
        """Get prompts from Hugging Face or local files"""
        try:
            from huggingface_hub import hf_hub_download
            
            # Download prompt files from HF if not available locally
            prompt_a_path = hf_hub_download(
                repo_id="sesame/csm-1b",
                filename="prompts/conversational_a.wav"
            )
            prompt_b_path = hf_hub_download(
                repo_id="sesame/csm-1b",
                filename="prompts/conversational_b.wav"
            )
            
            # Default prompts text
            prompt_a_text = (
                "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                "start really early I'd be like okay I'm gonna start revising now and then like "
                "you're revising for ages and then I just like start losing steam I didn't do that "
                "for the exam we had recently to be fair that was a more of a last minute scenario "
                "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                "sort of start the day with this not like a panic but like a"
            )
            
            prompt_b_text = (
                "like a super Mario level. Like it's very like high detail. And like, once you get "
                "into the park, it just like, everything looks like a computer game and they have all "
                "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                "will have like a question block. And if you like, you know, punch it, a coin will "
                "come out. So like everyone, when they come into the park, they get like this little "
                "bracelet and then you can go punching question blocks around."
            )
            
            # Load and prepare the prompts
            def load_prompt_audio(path, sample_rate):
                audio, sr = torchaudio.load(path)
                if sr != sample_rate:
                    audio = torchaudio.functional.resample(
                        audio.squeeze(0), orig_freq=sr, new_freq=sample_rate
                    )
                else:
                    audio = audio.squeeze(0)
                return audio
            
            prompt_a_audio = load_prompt_audio(prompt_a_path, self.generator.sample_rate)
            prompt_b_audio = load_prompt_audio(prompt_b_path, self.generator.sample_rate)
            
            prompt_a = Segment(text=prompt_a_text, speaker=0, audio=prompt_a_audio)
            prompt_b = Segment(text=prompt_b_text, speaker=1, audio=prompt_b_audio)
            
            return [prompt_a, prompt_b]
            
        except Exception as e:
            print(f"Failed to load prompt segments: {e}")
            # Return empty list on failure
            return []
    
    def run_basic_voice_tests(self):
        """Run basic voice generation tests with different types of therapeutic statements"""
        if not self.config.run_basic_tests:
            return
            
        print("\n==== Running Basic Voice Tests ====")
        results = {}
        
        # Test different statement types
        for category, statements in THERAPEUTIC_STATEMENTS.items():
            category_results = []
            
            print(f"\nTesting {category} statements...")
            for i, statement in enumerate(statements):
                if i > 1 and not self.config.test_all:  # Only test 2 statements per category if not test_all
                    break
                    
                print(f"  Generating: '{statement[:50]}...'")
                test_results = {}
                
                # Run multiple times to get average performance
                for run in range(self.config.num_runs):
                    start_time = time.time()
                    
                    try:
                        # Generate audio with speaker 0 (therapist)
                        audio = self.generator.generate(
                            text=statement,
                            speaker=0,
                            context=self.get_prompt_segments(),
                            max_audio_length_ms=30000,
                            temperature=0.9
                        )
                        
                        generation_time = time.time() - start_time
                        audio_length_s = len(audio) / self.generator.sample_rate
                        rtf = generation_time / audio_length_s  # Real-time factor
                        
                        test_results[f"run_{run}"] = {
                            "generation_time_s": generation_time,
                            "audio_length_s": audio_length_s,
                            "real_time_factor": rtf
                        }
                        
                        # Save audio file (only on first run)
                        if run == 0 and self.config.save_audio:
                            # Create a filename-safe version of the statement
                            safe_name = f"{category}_{i}"
                            output_path = os.path.join(self.config.output_dir, "basic_tests", f"{safe_name}.wav")
                            torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                            test_results["audio_path"] = output_path
                            
                    except Exception as e:
                        print(f"  Error generating audio: {e}")
                        test_results[f"run_{run}"] = {"error": str(e)}
                
                # Calculate averages across runs
                times = [r.get("generation_time_s", 0) for r in test_results.values() if isinstance(r, dict) and "error" not in r]
                lengths = [r.get("audio_length_s", 0) for r in test_results.values() if isinstance(r, dict) and "error" not in r]
                rtfs = [r.get("real_time_factor", 0) for r in test_results.values() if isinstance(r, dict) and "error" not in r]
                
                if times and lengths and rtfs:
                    summary = {
                        "statement": statement,
                        "avg_generation_time_s": sum(times) / len(times),
                        "avg_audio_length_s": sum(lengths) / len(lengths),
                        "avg_real_time_factor": sum(rtfs) / len(rtfs),
                        "word_count": len(statement.split()),
                        "detail": test_results
                    }
                    
                    print(f"  Results: {summary['avg_generation_time_s']:.2f}s generation, "
                          f"{summary['avg_audio_length_s']:.2f}s audio, "
                          f"RTF: {summary['avg_real_time_factor']:.2f}")
                    
                    category_results.append(summary)
            
            results[category] = category_results
        
        self.results["basic_tests"] = results
        self._save_results("basic_tests")
        
        return results
    
    def run_context_tests(self):
        """Test voice generation with conversation context"""
        if not self.config.run_context_tests:
            return
            
        print("\n==== Running Context Tests ====")
        results = {}
        
        # Generate audio for conversation pairs
        for i, pair in enumerate(CONTEXT_PAIRS):
            print(f"\nTesting context pair {i+1}...")
            test_results = {}
            
            try:
                # Create base prompt segments
                base_segments = self.get_prompt_segments()
                
                # First, generate client speech
                client_statement = pair["client"]
                print(f"  Generating client: '{client_statement[:50]}...'")
                
                client_audio = self.generator.generate(
                    text=client_statement,
                    speaker=1,  # Client is speaker 1
                    context=base_segments,
                    max_audio_length_ms=20000,
                )
                
                # Add client segment to context
                client_segment = Segment(
                    text=client_statement,
                    speaker=1,
                    audio=client_audio
                )
                context_segments = base_segments + [client_segment]
                
                # Then generate therapist response
                therapist_statement = pair["therapist"]
                print(f"  Generating therapist: '{therapist_statement[:50]}...'")
                
                start_time = time.time()
                therapist_audio = self.generator.generate(
                    text=therapist_statement,
                    speaker=0,  # Therapist is speaker 0
                    context=context_segments,
                    max_audio_length_ms=30000,
                )
                
                generation_time = time.time() - start_time
                audio_length_s = len(therapist_audio) / self.generator.sample_rate
                
                # Save the conversation audio
                if self.config.save_audio:
                    # Save individual utterances
                    client_path = os.path.join(self.config.output_dir, "context_tests", f"context_{i}_client.wav")
                    therapist_path = os.path.join(self.config.output_dir, "context_tests", f"context_{i}_therapist.wav")
                    
                    torchaudio.save(client_path, client_audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                    torchaudio.save(therapist_path, therapist_audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                    
                    # Concatenate for full conversation
                    # Add a short silence between utterances
                    silence_len = int(0.5 * self.generator.sample_rate)
                    silence = torch.zeros(silence_len)
                    full_audio = torch.cat([client_audio, silence, therapist_audio])
                    full_path = os.path.join(self.config.output_dir, "context_tests", f"context_{i}_full.wav")
                    torchaudio.save(full_path, full_audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                
                test_results = {
                    "client_statement": client_statement,
                    "therapist_statement": therapist_statement,
                    "generation_time_s": generation_time,
                    "audio_length_s": audio_length_s,
                    "real_time_factor": generation_time / audio_length_s,
                    "client_audio_path": client_path if self.config.save_audio else None,
                    "therapist_audio_path": therapist_path if self.config.save_audio else None,
                    "full_audio_path": full_path if self.config.save_audio else None
                }
                
                print(f"  Results: {test_results['generation_time_s']:.2f}s generation, "
                      f"{test_results['audio_length_s']:.2f}s audio, "
                      f"RTF: {test_results['real_time_factor']:.2f}")
                
            except Exception as e:
                print(f"  Error in context test {i+1}: {e}")
                test_results = {"error": str(e)}
            
            results[f"context_pair_{i}"] = test_results
        
        self.results["context_tests"] = results
        self._save_results("context_tests")
        
        return results
    
    def run_quality_tests(self):
        """Run audio quality evaluation tests"""
        if not self.config.run_quality_tests:
            return
            
        print("\n==== Running Quality Tests ====")
        results = {}
        
        # Select a sample of statements for quality testing
        test_statements = [
            ("emotional", THERAPEUTIC_STATEMENTS["emotional"][0]),
            ("medium", THERAPEUTIC_STATEMENTS["medium"][1]),
            ("long", THERAPEUTIC_STATEMENTS["long"][0])
        ]
        
        for category, statement in test_statements:
            print(f"\nQuality testing: {category}")
            print(f"  Statement: '{statement[:50]}...'")
            
            try:
                # Generate audio
                audio = self.generator.generate(
                    text=statement,
                    speaker=0,
                    context=self.get_prompt_segments(),
                    max_audio_length_ms=30000,
                )
                
                # Calculate audio quality metrics
                metrics = self._calculate_audio_metrics(audio)
                
                # Save audio for manual evaluation
                if self.config.save_audio:
                    output_path = os.path.join(self.config.output_dir, "quality_tests", f"quality_{category}.wav")
                    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                    metrics["audio_path"] = output_path
                
                print(f"  Quality metrics: SNR={metrics['snr']:.2f}dB, "
                      f"Dynamic Range={metrics['dynamic_range']:.2f}dB")
                
                results[category] = {
                    "statement": statement,
                    "metrics": metrics
                }
                
            except Exception as e:
                print(f"  Error in quality test: {e}")
                results[category] = {"error": str(e)}
        
        self.results["quality_tests"] = results
        self._save_results("quality_tests")
        
        return results
    
    def _calculate_audio_metrics(self, audio: torch.Tensor) -> Dict[str, float]:
        """Calculate objective audio quality metrics"""
        # Convert to numpy for calculations
        audio_np = audio.cpu().numpy()
        
        # Calculate signal power
        signal_power = np.mean(audio_np ** 2)
        signal_db = 10 * np.log10(signal_power) if signal_power > 0 else -100
        
        # Estimate noise floor (using the lowest 10% of non-zero frames)
        frame_size = 1024
        frames = []
        for i in range(0, len(audio_np) - frame_size, frame_size):
            frame = audio_np[i:i+frame_size]
            frame_power = np.mean(frame ** 2)
            if frame_power > 0:
                frames.append(frame_power)
        
        frames.sort()
        if frames:
            noise_floor = np.mean(frames[:max(1, len(frames) // 10)])
            noise_db = 10 * np.log10(noise_floor) if noise_floor > 0 else -100
        else:
            noise_db = -100
            
        # SNR calculation
        snr = signal_db - noise_db
        
        # Dynamic range
        if len(audio_np) > 0:
            dynamic_range = 20 * np.log10(np.max(np.abs(audio_np)) / (np.mean(np.abs(audio_np)) + 1e-10))
        else:
            dynamic_range = 0
            
        # Zero crossing rate (rough measure of voice quality/noise)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_np))))
        zcr = zero_crossings / len(audio_np)
        
        return {
            "signal_db": float(signal_db),
            "noise_db": float(noise_db),
            "snr": float(snr),
            "dynamic_range": float(dynamic_range),
            "zero_crossing_rate": float(zcr)
        }
    
    def run_performance_tests(self):
        """Run performance benchmarking tests"""
        if not self.config.run_performance_tests:
            return
            
        print("\n==== Running Performance Tests ====")
        results = {}
        
        # Test different statement lengths
        test_statements = [
            ("short", THERAPEUTIC_STATEMENTS["short"][0]),
            ("medium", THERAPEUTIC_STATEMENTS["medium"][0]),
            ("long", THERAPEUTIC_STATEMENTS["long"][0])
        ]
        
        # Test different temperature values
        temperatures = [0.7, 0.9, 1.1]
        
        # Cold start vs warm start
        print("\nTesting cold start vs warm start...")
        
        # Cold start
        self.generator.reset_caches()
        torch.cuda.empty_cache()
        
        cold_start = time.time()
        cold_audio = self.generator.generate(
            text=test_statements[1][1],  # Medium statement
            speaker=0,
            context=self.get_prompt_segments(),
            max_audio_length_ms=10000,
        )
        cold_time = time.time() - cold_start
        
        # Warm start
        warm_start = time.time()
        warm_audio = self.generator.generate(
            text=test_statements[1][1],  # Same statement
            speaker=0,
            context=self.get_prompt_segments(),
            max_audio_length_ms=10000,
        )
        warm_time = time.time() - warm_start
        
        print(f"  Cold start: {cold_time:.2f}s, Warm start: {warm_time:.2f}s")
        results["cold_vs_warm"] = {
            "cold_start_time": cold_time,
            "warm_start_time": warm_time,
            "speedup_factor": cold_time / warm_time if warm_time > 0 else 0
        }
        
        # Test statement lengths
        print("\nTesting different statement lengths...")
        length_results = {}
        
        for category, statement in test_statements:
            print(f"  Testing {category} statement...")
            run_results = []
            
            for run in range(self.config.num_runs):
                start_time = time.time()
                
                audio = self.generator.generate(
                    text=statement,
                    speaker=0,
                    context=self.get_prompt_segments(),
                    max_audio_length_ms=30000,
                )
                
                generation_time = time.time() - start_time
                audio_length = len(audio) / self.generator.sample_rate
                words_per_second = len(statement.split()) / audio_length
                
                run_results.append({
                    "generation_time_s": generation_time,
                    "audio_length_s": audio_length,
                    "real_time_factor": generation_time / audio_length,
                    "words_per_second": words_per_second
                })
            
            # Calculate averages
            avg_gen_time = sum(r["generation_time_s"] for r in run_results) / len(run_results)
            avg_audio_len = sum(r["audio_length_s"] for r in run_results) / len(run_results)
            avg_rtf = sum(r["real_time_factor"] for r in run_results) / len(run_results)
            avg_wps = sum(r["words_per_second"] for r in run_results) / len(run_results)
            
            print(f"    Avg RTF: {avg_rtf:.2f}, WPS: {avg_wps:.2f}")
            
            length_results[category] = {
                "statement": statement,
                "word_count": len(statement.split()),
                "avg_generation_time_s": avg_gen_time,
                "avg_audio_length_s": avg_audio_len,
                "avg_real_time_factor": avg_rtf,
                "avg_words_per_second": avg_wps,
                "runs": run_results
            }
        
        results["statement_lengths"] = length_results
        
        # Test temperature impact
        if self.config.test_all:
            print("\nTesting different temperature values...")
            temp_results = {}
            
            statement = test_statements[1][1]  # Medium statement
            
            for temp in temperatures:
                print(f"  Testing temperature {temp}...")
                run_results = []
                
                for run in range(self.config.num_runs):
                    start_time = time.time()
                    
                    audio = self.generator.generate(
                        text=statement,
                        speaker=0,
                        context=self.get_prompt_segments(),
                        max_audio_length_ms=15000,
                        temperature=temp
                    )
                    
                    generation_time = time.time() - start_time
                    audio_length = len(audio) / self.generator.sample_rate
                    
                    run_results.append({
                        "generation_time_s": generation_time,
                        "audio_length_s": audio_length,
                        "real_time_factor": generation_time / audio_length
                    })
                    
                    # Save sample for temperature comparison
                    if run == 0 and self.config.save_audio:
                        output_path = os.path.join(self.config.output_dir, "performance_tests", f"temp_{temp}.wav")
                        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                
                # Calculate averages
                avg_gen_time = sum(r["generation_time_s"] for r in run_results) / len(run_results)
                avg_audio_len = sum(r["audio_length_s"] for r in run_results) / len(run_results)
                avg_rtf = sum(r["real_time_factor"] for r in run_results) / len(run_results)
                
                print(f"    Avg RTF: {avg_rtf:.2f}")
                
                temp_results[f"temp_{temp}"] = {
                    "temperature": temp,
                    "avg_generation_time_s": avg_gen_time,
                    "avg_audio_length_s": avg_audio_len,
                    "avg_real_time_factor": avg_rtf,
                    "runs": run_results
                }
            
            results["temperature_impact"] = temp_results
        
        self.results["performance_tests"] = results
        self._save_results("performance_tests")
        
        return results
    
    def run_edge_case_tests(self):
        """Run edge case tests"""
        if not self.config.run_edge_tests:
            return
            
        print("\n==== Running Edge Case Tests ====")
        results = {}
        
        # Test edge cases
        for category, statements in EDGE_CASES.items():
            print(f"\nTesting {category} edge cases...")
            category_results = []
            
            for i, statement in enumerate(statements):
                if i > 0 and not self.config.test_all:
                    break
                    
                print(f"  Generating: '{statement[:50]}...'")
                
                try:
                    # Generate audio
                    start_time = time.time()
                    audio = self.generator.generate(
                        text=statement,
                        speaker=0,
                        context=self.get_prompt_segments(),
                        max_audio_length_ms=30000,
                    )
                    
                    generation_time = time.time() - start_time
                    audio_length = len(audio) / self.generator.sample_rate
                    
                    # Save audio
                    if self.config.save_audio:
                        output_path = os.path.join(self.config.output_dir, "edge_tests", f"{category}_{i}.wav")
                        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                    
                    test_result = {
                        "statement": statement,
                        "generation_time_s": generation_time,
                        "audio_length_s": audio_length,
                        "real_time_factor": generation_time / audio_length,
                        "success": True
                    }
                    
                    print(f"  Success: {generation_time:.2f}s generation, {audio_length:.2f}s audio")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    test_result = {
                        "statement": statement,
                        "error": str(e),
                        "success": False
                    }
                
                category_results.append(test_result)
            
            results[category] = category_results
        
        # Test maximum length
        print("\nTesting maximum length handling...")
        
        # Concatenate all long statements to create very long input
        very_long = " ".join(THERAPEUTIC_STATEMENTS["long"])
        
        try:
            # Try generating with very long input
            start_time = time.time()
            audio = self.generator.generate(
                text=very_long,
                speaker=0,
                context=self.get_prompt_segments(),
                max_audio_length_ms=60000,  # Longer max time
            )
            
            generation_time = time.time() - start_time
            audio_length = len(audio) / self.generator.sample_rate
            
            if self.config.save_audio:
                output_path = os.path.join(self.config.output_dir, "edge_tests", "max_length.wav")
                torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            
            results["max_length"] = {
                "word_count": len(very_long.split()),
                "character_count": len(very_long),
                "generation_time_s": generation_time,
                "audio_length_s": audio_length,
                "real_time_factor": generation_time / audio_length,
                "success": True
            }
            
            print(f"  Max length test succeeded: {audio_length:.2f}s audio generated")
            
        except Exception as e:
            print(f"  Max length test failed: {e}")
            results["max_length"] = {
                "word_count": len(very_long.split()),
                "character_count": len(very_long),
                "error": str(e),
                "success": False
            }
        
        self.results["edge_tests"] = results
        self._save_results("edge_tests")
        
        return results
    
    def _save_results(self, test_type: str):
        """Save test results to JSON file"""
        output_path = os.path.join(self.config.output_dir, f"{test_type}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results[test_type], f, indent=2)
            
        print(f"Saved {test_type} results to {output_path}")
    
    def plot_results(self):
        """Plot test results"""
        if not self.config.plot_results:
            return
            
        print("\n==== Plotting Results ====")
        
        # Plot basic test results
        if "basic_tests" in self.results and self.results["basic_tests"]:
            self._plot_basic_results()
            
        # Plot performance results
        if "performance_tests" in self.results and self.results["performance_tests"]:
            self._plot_performance_results()
    
    def _plot_basic_results(self):
        """Plot basic test results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Gather data for categories
            categories = []
            rtfs = []
            lengths = []
            generation_times = []
            words = []
            
            for category, results in self.results["basic_tests"].items():
                for result in results:
                    categories.append(category)
                    rtfs.append(result.get("avg_real_time_factor", 0))
                    lengths.append(result.get("avg_audio_length_s", 0))
                    generation_times.append(result.get("avg_generation_time_s", 0))
                    words.append(result.get("word_count", 0))
            
            # Plot Real-Time Factor by category
            plt.subplot(2, 2, 1)
            category_set = set(categories)
            for cat in category_set:
                indices = [i for i, x in enumerate(categories) if x == cat]
                cat_rtfs = [rtfs[i] for i in indices]
                plt.scatter([cat] * len(cat_rtfs), cat_rtfs, label=cat)
            
            plt.title('Real-Time Factor by Statement Type')
            plt.ylabel('Real-Time Factor')
            plt.grid(True, alpha=0.3)
            
            # Plot Generation Time vs. Audio Length
            plt.subplot(2, 2, 2)
            for cat in category_set:
                indices = [i for i, x in enumerate(categories) if x == cat]
                x_vals = [lengths[i] for i in indices]
                y_vals = [generation_times[i] for i in indices]
                plt.scatter(x_vals, y_vals, label=cat)
                
            plt.title('Generation Time vs. Audio Length')
            plt.xlabel('Audio Length (s)')
            plt.ylabel('Generation Time (s)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot Word Count vs. Audio Length
            plt.subplot(2, 2, 3)
            for cat in category_set:
                indices = [i for i, x in enumerate(categories) if x == cat]
                x_vals = [words[i] for i in indices]
                y_vals = [lengths[i] for i in indices]
                plt.scatter(x_vals, y_vals, label=cat)
                
            plt.title('Word Count vs. Audio Length')
            plt.xlabel('Word Count')
            plt.ylabel('Audio Length (s)')
            plt.grid(True, alpha=0.3)
            
            # Plot Word Count vs. RTF
            plt.subplot(2, 2, 4)
            plt.scatter(words, rtfs)
            plt.title('Word Count vs. RTF')
            plt.xlabel('Word Count')
            plt.ylabel('Real-Time Factor')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.config.output_dir, "basic_tests_plot.png")
            plt.savefig(plot_path)
            print(f"Basic test results plot saved to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error plotting basic results: {e}")
    
    def _plot_performance_results(self):
        """Plot performance test results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Cold vs. Warm Start
            if "cold_vs_warm" in self.results["performance_tests"]:
                cold_warm = self.results["performance_tests"]["cold_vs_warm"]
                plt.subplot(2, 2, 1)
                plt.bar(["Cold Start", "Warm Start"], 
                       [cold_warm.get("cold_start_time", 0), cold_warm.get("warm_start_time", 0)])
                plt.title('Cold Start vs. Warm Start')
                plt.ylabel('Time (s)')
                plt.grid(True, alpha=0.3)
            
            # Statement Lengths
            if "statement_lengths" in self.results["performance_tests"]:
                lengths = self.results["performance_tests"]["statement_lengths"]
                categories = list(lengths.keys())
                rtfs = [lengths[cat].get("avg_real_time_factor", 0) for cat in categories]
                gen_times = [lengths[cat].get("avg_generation_time_s", 0) for cat in categories]
                word_counts = [lengths[cat].get("word_count", 0) for cat in categories]
                
                plt.subplot(2, 2, 2)
                plt.bar(categories, rtfs)
                plt.title('Real-Time Factor by Statement Length')
                plt.ylabel('Real-Time Factor')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.bar(categories, gen_times)
                plt.title('Generation Time by Statement Length')
                plt.ylabel('Generation Time (s)')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.scatter(word_counts, rtfs)
                for i, cat in enumerate(categories):
                    plt.annotate(cat, (word_counts[i], rtfs[i]))
                plt.title('Word Count vs. RTF')
                plt.xlabel('Word Count')
                plt.ylabel('Real-Time Factor')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.config.output_dir, "performance_tests_plot.png")
            plt.savefig(plot_path)
            print(f"Performance test results plot saved to {plot_path}")
            
            # Temperature impact plot if available
            if "temperature_impact" in self.results["performance_tests"]:
                temp_impact = self.results["performance_tests"]["temperature_impact"]
                
                plt.figure(figsize=(10, 6))
                temps = [float(k.split('_')[1]) for k in temp_impact.keys()]
                rtfs = [temp_impact[f"temp_{t}"].get("avg_real_time_factor", 0) for t in temps]
                
                plt.bar([str(t) for t in temps], rtfs)
                plt.title('Temperature Impact on Real-Time Factor')
                plt.xlabel('Temperature')
                plt.ylabel('Real-Time Factor')
                plt.grid(True, alpha=0.3)
                
                # Save temperature plot
                temp_plot_path = os.path.join(self.config.output_dir, "temperature_impact_plot.png")
                plt.savefig(temp_plot_path)
                print(f"Temperature impact plot saved to {temp_plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error plotting performance results: {e}")
    
    def run_all_tests(self):
        """Run all test suites"""
        if not self.initialize_model():
            print("Model initialization failed. Cannot run tests.")
            return False
        
        # Run all test suites
        self.run_basic_voice_tests()
        self.run_context_tests()
        self.run_quality_tests()
        self.run_performance_tests()
        self.run_edge_case_tests()
        
        # Plot results
        self.plot_results()
        
        print("\n==== All Tests Completed ====")
        print(f"Results saved to {self.config.output_dir}")
        
        return True

# Run the tests if executed directly
if __name__ == "__main__":
    # Create test configuration
    config = TestConfig(
        output_dir="/content/drive/MyDrive/psychoanalyst_assistant/test_results",
        num_runs=3,
        test_all=True,
        save_audio=True,
        plot_results=True
    )
    
    # Initialize and run tests
    test_runner = CSMVoiceTest(config)
    test_runner.run_all_tests()