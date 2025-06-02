#!/usr/bin/env python3
"""
Content Filter CLI Command

Add this to your CLI handlers for easy content filter testing.
"""
import click
from beautyai_inference.services.inference.content_filter_service import ContentFilterService


@click.command()
@click.option('--text', '-t', help='Text to test against content filter')
@click.option('--language', '-l', default='ar', help='Language (ar/en)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--stats', '-s', is_flag=True, help='Show filter statistics')
def test_filter(text, language, interactive, stats):
    """Test the content filter with given text."""
    
    filter_service = ContentFilterService()
    
    if stats:
        stats_data = filter_service.get_filter_stats()
        click.echo(f"📊 Content Filter Statistics:")
        click.echo(f"   Forbidden Topics: {stats_data['total_forbidden_topics']}")
        click.echo(f"   Forbidden Keywords: {stats_data['total_forbidden_keywords']}")
        click.echo(f"   Question Patterns: {stats_data['total_question_patterns']}")
        click.echo(f"   CSV File Exists: {stats_data['csv_exists']}")
        return
    
    if interactive:
        click.echo("🔄 Interactive Content Filter Testing")
        click.echo("Type 'quit' to exit.")
        
        while True:
            try:
                user_input = click.prompt("\n👤 Enter text to test", type=str).strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("👋 Goodbye!")
                    break
                
                result = filter_service.filter_content(user_input, language=language)
                
                if result.is_allowed:
                    click.echo("✅ ALLOWED - Content would be processed")
                else:
                    click.echo("🚫 BLOCKED - Content filter activated")
                    click.echo(f"   Reason: {result.filter_reason}")
                    click.echo(f"   Confidence: {result.confidence_score:.2f}")
                    if result.matched_patterns:
                        click.echo(f"   Matched: {result.matched_patterns[:3]}")
                
            except click.Abort:
                click.echo("\n👋 Goodbye!")
                break
    
    elif text:
        result = filter_service.filter_content(text, language=language)
        
        if result.is_allowed:
            click.echo("✅ ALLOWED")
        else:
            click.echo("🚫 BLOCKED")
            click.echo(f"Reason: {result.filter_reason}")
            click.echo(f"Confidence: {result.confidence_score:.2f}")
            if result.suggested_response:
                click.echo(f"Response: {result.suggested_response}")
    
    else:
        click.echo("❌ Please provide text with --text or use --interactive mode")
        click.echo("Example: beautyai test-filter --text 'ما تكلفة البوتوكس؟'")


if __name__ == '__main__':
    test_filter()
