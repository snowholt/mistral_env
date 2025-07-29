#!/usr/bin/env python3
"""
Benchmark Results Analyzer for BeautyAI
=======================================

This script analyzes results from the enhanced benchmarking system and generates
comprehensive reports, visualizations, and insights for model comparison.

Usage:
    python analyze_benchmark_results.py results.json
    python analyze_benchmark_results.py results.json --generate-report
    python analyze_benchmark_results.py results.json --export-csv --export-html
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkAnalyzer:
    """Analyzer for BeautyAI benchmark results."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to JSON results file
        """
        self.results_file = Path(results_file)
        self.data = self._load_results()
        self.analysis = {}
        
    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded benchmark results from {self.results_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def analyze_content_filtering_effectiveness(self) -> Dict[str, Any]:
        """Analyze content filtering effectiveness across models."""
        results = {}
        
        # Extract model results
        if 'multi_model_comparison' in self.data:
            model_results = self.data['multi_model_comparison']['model_results']
        elif 'single_model_results' in self.data:
            model_name = self.data['benchmark_config']['model_name']
            model_results = {model_name: self.data['single_model_results']}
        else:
            logger.warning("No model results found in data")
            return {}
        
        for model_name, model_data in model_results.items():
            if 'summary' not in model_data:
                continue
                
            summary = model_data['summary']
            
            results[model_name] = {
                'total_questions': summary.get('total_questions', 0),
                'content_filtered_count': summary.get('content_filtered_responses', 0),
                'content_filter_rate': summary.get('content_filter_rate_percent', 0),
                'success_rate': summary.get('success_rate_percent', 0),
                'filtering_effectiveness': 'High' if summary.get('content_filter_rate_percent', 0) > 80 else 
                                         'Medium' if summary.get('content_filter_rate_percent', 0) > 50 else 'Low'
            }
        
        # Calculate overall statistics
        total_questions = sum(r['total_questions'] for r in results.values())
        total_filtered = sum(r['content_filtered_count'] for r in results.values())
        overall_filter_rate = (total_filtered / total_questions * 100) if total_questions > 0 else 0
        
        analysis = {
            'model_analysis': results,
            'overall_statistics': {
                'total_questions_tested': total_questions,
                'total_content_filtered': total_filtered,
                'overall_filter_rate_percent': round(overall_filter_rate, 2),
                'models_with_high_filtering': len([r for r in results.values() if r['filtering_effectiveness'] == 'High']),
                'models_with_low_filtering': len([r for r in results.values() if r['filtering_effectiveness'] == 'Low'])
            }
        }
        
        self.analysis['content_filtering'] = analysis
        return analysis
    
    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics across models."""
        results = {}
        
        # Extract model results
        if 'multi_model_comparison' in self.data:
            model_results = self.data['multi_model_comparison']['model_results']
        elif 'single_model_results' in self.data:
            model_name = self.data['benchmark_config']['model_name']
            model_results = {model_name: self.data['single_model_results']}
        else:
            return {}
        
        latencies = []
        throughputs = []
        
        for model_name, model_data in model_results.items():
            if 'summary' not in model_data:
                continue
                
            summary = model_data['summary']
            
            avg_latency = summary.get('average_latency_ms', 0)
            throughput = summary.get('questions_per_second', 0)
            
            results[model_name] = {
                'average_latency_ms': avg_latency,
                'questions_per_second': throughput,
                'success_rate': summary.get('success_rate_percent', 0),
                'total_duration': summary.get('total_duration_seconds', 0),
                'performance_grade': self._calculate_performance_grade(avg_latency, throughput)
            }
            
            latencies.append(avg_latency)
            throughputs.append(throughput)
        
        # Calculate statistics
        analysis = {
            'model_analysis': results,
            'overall_statistics': {
                'fastest_model': min(results.keys(), key=lambda x: results[x]['average_latency_ms']) if results else None,
                'highest_throughput_model': max(results.keys(), key=lambda x: results[x]['questions_per_second']) if results else None,
                'average_latency_ms': round(np.mean(latencies), 2) if latencies else 0,
                'average_throughput': round(np.mean(throughputs), 2) if throughputs else 0,
                'latency_std': round(np.std(latencies), 2) if latencies else 0,
                'throughput_std': round(np.std(throughputs), 2) if throughputs else 0
            }
        }
        
        self.analysis['performance'] = analysis
        return analysis
    
    def _calculate_performance_grade(self, latency: float, throughput: float) -> str:
        """Calculate performance grade based on latency and throughput."""
        # Grading criteria (can be adjusted based on requirements)
        if latency < 500 and throughput > 5:
            return 'A'
        elif latency < 1000 and throughput > 3:
            return 'B'
        elif latency < 2000 and throughput > 1:
            return 'C'
        else:
            return 'D'
    
    def analyze_response_quality(self) -> Dict[str, Any]:
        """Analyze response quality patterns."""
        results = {}
        
        # Extract model results
        if 'multi_model_comparison' in self.data:
            model_results = self.data['multi_model_comparison']['model_results']
        elif 'single_model_results' in self.data:
            model_name = self.data['benchmark_config']['model_name']
            model_results = {model_name: self.data['single_model_results']}
        else:
            return {}
        
        for model_name, model_data in model_results.items():
            if 'successful_results' not in model_data:
                continue
            
            successful_results = model_data['successful_results']
            response_lengths = []
            filtered_responses = 0
            
            for result in successful_results:
                response = result.get('response', '')
                response_lengths.append(len(response))
                
                if result.get('content_filtered', False):
                    filtered_responses += 1
            
            if response_lengths:
                results[model_name] = {
                    'average_response_length': round(np.mean(response_lengths), 2),
                    'median_response_length': round(np.median(response_lengths), 2),
                    'response_length_std': round(np.std(response_lengths), 2),
                    'min_response_length': min(response_lengths),
                    'max_response_length': max(response_lengths),
                    'filtered_responses': filtered_responses,
                    'total_responses': len(successful_results)
                }
        
        self.analysis['response_quality'] = results
        return results
    
    def generate_visualizations(self, output_dir: str = "benchmark_analysis"):
        """Generate visualization charts for the analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Ensure we have analysis data
        if not self.analysis:
            self.run_complete_analysis()
        
        # 1. Content Filtering Effectiveness Chart
        if 'content_filtering' in self.analysis:
            self._plot_content_filtering_effectiveness(output_path)
        
        # 2. Performance Comparison Chart
        if 'performance' in self.analysis:
            self._plot_performance_comparison(output_path)
        
        # 3. Response Quality Distribution
        if 'response_quality' in self.analysis:
            self._plot_response_quality(output_path)
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_content_filtering_effectiveness(self, output_path: Path):
        """Plot content filtering effectiveness chart."""
        try:
            data = self.analysis['content_filtering']['model_analysis']
            
            models = list(data.keys())
            filter_rates = [data[model]['content_filter_rate'] for model in models]
            success_rates = [data[model]['success_rate'] for model in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Content filter rates
            bars1 = ax1.bar(models, filter_rates, color='coral', alpha=0.7)
            ax1.set_title('Content Filtering Effectiveness by Model')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Content Filter Rate (%)')
            ax1.set_ylim(0, 100)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, rate in zip(bars1, filter_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            # Success rates vs filter rates
            ax2.scatter(filter_rates, success_rates, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax2.annotate(model, (filter_rates[i], success_rates[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax2.set_xlabel('Content Filter Rate (%)')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Content Filtering vs Success Rate')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'content_filtering_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting content filtering chart: {e}")
    
    def _plot_performance_comparison(self, output_path: Path):
        """Plot performance comparison chart."""
        try:
            data = self.analysis['performance']['model_analysis']
            
            models = list(data.keys())
            latencies = [data[model]['average_latency_ms'] for model in models]
            throughputs = [data[model]['questions_per_second'] for model in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Latency comparison
            bars1 = ax1.bar(models, latencies, color='lightblue', alpha=0.7)
            ax1.set_title('Average Latency by Model')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Average Latency (ms)')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, latency in zip(bars1, latencies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latencies) * 0.01,
                        f'{latency:.0f}ms', ha='center', va='bottom')
            
            # Throughput comparison
            bars2 = ax2.bar(models, throughputs, color='lightgreen', alpha=0.7)
            ax2.set_title('Throughput by Model')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Questions per Second')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, throughput in zip(bars2, throughputs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs) * 0.01,
                        f'{throughput:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance chart: {e}")
    
    def _plot_response_quality(self, output_path: Path):
        """Plot response quality analysis."""
        try:
            data = self.analysis['response_quality']
            
            models = list(data.keys())
            avg_lengths = [data[model]['average_response_length'] for model in models]
            length_stds = [data[model]['response_length_std'] for model in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Average response length
            bars1 = ax1.bar(models, avg_lengths, color='gold', alpha=0.7)
            ax1.set_title('Average Response Length by Model')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Average Response Length (characters)')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Response length variability
            bars2 = ax2.bar(models, length_stds, color='orchid', alpha=0.7)
            ax2.set_title('Response Length Variability by Model')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Standard Deviation (characters)')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(output_path / 'response_quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting response quality chart: {e}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete analysis on the benchmark results."""
        logger.info("Running complete benchmark analysis...")
        
        # Run all analysis modules
        self.analyze_content_filtering_effectiveness()
        self.analyze_performance_metrics()
        self.analyze_response_quality()
        
        return self.analysis
    
    def generate_text_report(self, output_file: str = None) -> str:
        """Generate comprehensive text report."""
        if not self.analysis:
            self.run_complete_analysis()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BEAUTYAI BENCHMARK ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Source File: {self.results_file}")
        report_lines.append("")
        
        # Content Filtering Analysis
        if 'content_filtering' in self.analysis:
            report_lines.append("1. CONTENT FILTERING EFFECTIVENESS")
            report_lines.append("-" * 40)
            
            cf_data = self.analysis['content_filtering']
            overall = cf_data['overall_statistics']
            
            report_lines.append(f"Total Questions Tested: {overall['total_questions_tested']}")
            report_lines.append(f"Total Content Filtered: {overall['total_content_filtered']}")
            report_lines.append(f"Overall Filter Rate: {overall['overall_filter_rate_percent']:.1f}%")
            report_lines.append(f"Models with High Filtering (>80%): {overall['models_with_high_filtering']}")
            report_lines.append(f"Models with Low Filtering (<50%): {overall['models_with_low_filtering']}")
            report_lines.append("")
            
            report_lines.append("Model-by-Model Content Filtering:")
            for model, data in cf_data['model_analysis'].items():
                report_lines.append(f"  {model}:")
                report_lines.append(f"    Filter Rate: {data['content_filter_rate']:.1f}%")
                report_lines.append(f"    Success Rate: {data['success_rate']:.1f}%")
                report_lines.append(f"    Effectiveness: {data['filtering_effectiveness']}")
            report_lines.append("")
        
        # Performance Analysis
        if 'performance' in self.analysis:
            report_lines.append("2. PERFORMANCE ANALYSIS")
            report_lines.append("-" * 40)
            
            perf_data = self.analysis['performance']
            overall = perf_data['overall_statistics']
            
            report_lines.append(f"Fastest Model: {overall.get('fastest_model', 'N/A')}")
            report_lines.append(f"Highest Throughput Model: {overall.get('highest_throughput_model', 'N/A')}")
            report_lines.append(f"Average Latency: {overall['average_latency_ms']:.1f}ms")
            report_lines.append(f"Average Throughput: {overall['average_throughput']:.1f} questions/sec")
            report_lines.append("")
            
            report_lines.append("Model Performance Grades:")
            for model, data in perf_data['model_analysis'].items():
                report_lines.append(f"  {model}: Grade {data['performance_grade']}")
                report_lines.append(f"    Latency: {data['average_latency_ms']:.1f}ms")
                report_lines.append(f"    Throughput: {data['questions_per_second']:.1f} q/s")
            report_lines.append("")
        
        # Response Quality Analysis
        if 'response_quality' in self.analysis:
            report_lines.append("3. RESPONSE QUALITY ANALYSIS")
            report_lines.append("-" * 40)
            
            for model, data in self.analysis['response_quality'].items():
                report_lines.append(f"{model}:")
                report_lines.append(f"  Avg Response Length: {data['average_response_length']:.0f} chars")
                report_lines.append(f"  Response Variability: {data['response_length_std']:.0f} chars")
                report_lines.append(f"  Total Responses: {data['total_responses']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("4. RECOMMENDATIONS")
        report_lines.append("-" * 40)
        recommendations = self._generate_recommendations()
        for recommendation in recommendations:
            report_lines.append(f"• {recommendation}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Text report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save text report: {e}")
        
        return report_text
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if 'content_filtering' in self.analysis:
            cf_data = self.analysis['content_filtering']
            overall_rate = cf_data['overall_statistics']['overall_filter_rate_percent']
            
            if overall_rate < 80:
                recommendations.append("Consider strengthening content filtering mechanisms - current rate is below 80%")
            
            low_filtering_models = [
                model for model, data in cf_data['model_analysis'].items()
                if data['filtering_effectiveness'] == 'Low'
            ]
            
            if low_filtering_models:
                recommendations.append(f"Models with low filtering effectiveness need attention: {', '.join(low_filtering_models)}")
        
        if 'performance' in self.analysis:
            perf_data = self.analysis['performance']
            
            slow_models = [
                model for model, data in perf_data['model_analysis'].items()
                if data['average_latency_ms'] > 2000
            ]
            
            if slow_models:
                recommendations.append(f"Consider optimizing these slow models: {', '.join(slow_models)}")
        
        if not recommendations:
            recommendations.append("All models are performing within acceptable parameters")
        
        return recommendations
    
    def export_to_csv(self, output_file: str):
        """Export analysis results to CSV format."""
        try:
            if not self.analysis:
                self.run_complete_analysis()
            
            # Prepare data for CSV export
            rows = []
            
            for analysis_type, analysis_data in self.analysis.items():
                if analysis_type == 'content_filtering' and 'model_analysis' in analysis_data:
                    for model, data in analysis_data['model_analysis'].items():
                        rows.append({
                            'analysis_type': 'content_filtering',
                            'model_name': model,
                            'metric': 'filter_rate_percent',
                            'value': data['content_filter_rate']
                        })
                        rows.append({
                            'analysis_type': 'content_filtering',
                            'model_name': model,
                            'metric': 'success_rate_percent',
                            'value': data['success_rate']
                        })
                
                elif analysis_type == 'performance' and 'model_analysis' in analysis_data:
                    for model, data in analysis_data['model_analysis'].items():
                        rows.append({
                            'analysis_type': 'performance',
                            'model_name': model,
                            'metric': 'average_latency_ms',
                            'value': data['average_latency_ms']
                        })
                        rows.append({
                            'analysis_type': 'performance',
                            'model_name': model,
                            'metric': 'questions_per_second',
                            'value': data['questions_per_second']
                        })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            logger.info(f"Analysis exported to CSV: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise


def main():
    """Main function for the benchmark analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze BeautyAI benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'results_file',
        help='Path to the benchmark results JSON file'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive text report'
    )
    
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export analysis to CSV file'
    )
    
    parser.add_argument(
        '--visualizations',
        action='store_true',
        help='Generate visualization charts'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_analysis',
        help='Output directory for generated files (default: benchmark_analysis)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = BenchmarkAnalyzer(args.results_file)
        
        # Run complete analysis
        analysis_results = analyzer.run_complete_analysis()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate text report
        if args.generate_report:
            report_file = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_text = analyzer.generate_text_report(str(report_file))
            print("\nBENCHMARK ANALYSIS SUMMARY:")
            print("=" * 50)
            # Print first few lines of the report
            lines = report_text.split('\n')
            for line in lines[:20]:  # Show first 20 lines
                print(line)
            if len(lines) > 20:
                print("... (see full report in output file)")
        
        # Export to CSV
        if args.export_csv:
            csv_file = output_dir / args.export_csv if not Path(args.export_csv).is_absolute() else args.export_csv
            analyzer.export_to_csv(str(csv_file))
        
        # Generate visualizations
        if args.visualizations:
            analyzer.generate_visualizations(str(output_dir))
        
        # Print summary
        print(f"\nAnalysis completed successfully!")
        print(f"Output directory: {output_dir}")
        
        if 'content_filtering' in analysis_results:
            cf_stats = analysis_results['content_filtering']['overall_statistics']
            print(f"Overall content filter rate: {cf_stats['overall_filter_rate_percent']:.1f}%")
        
        if 'performance' in analysis_results:
            perf_stats = analysis_results['performance']['overall_statistics']
            print(f"Average latency: {perf_stats['average_latency_ms']:.1f}ms")
            print(f"Average throughput: {perf_stats['average_throughput']:.1f} questions/sec")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
