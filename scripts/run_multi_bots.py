#!/usr/bin/env python3
"""
Multi-Bot Launcher: Spawns multiple bot instances in parallel and coordinates them via Flask
"""
import os
import sys
import json
import time
import threading
import multiprocessing
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from packages.core.bot_manager import BotManager
from apps.bot.bot_pro import AggressiveGrowthBot, app


class BotLauncher:
    """Orchestrates multiple bot instances"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.bot_manager = BotManager(config_path)
        self.bot_processes = {}  # {bot_id: Process}
        self.bot_instances = {}  # {bot_id: bot_instance} - for shared memory mode
        self.is_running = False
    
    def launch_bot_instance(self, bot_id: str, bot_config: dict):
        """
        Launch a single bot instance in a separate process or thread
        
        Args:
            bot_id: Unique bot identifier
            bot_config: Bot configuration dictionary
        """
        try:
            print(f"\n🚀 Launching bot: {bot_id} ({bot_config.get('name', 'Unknown')})")
            print(f"   RSI Config: Period={bot_config['rsi_config']['period']}, "
                  f"Oversold={bot_config['rsi_config']['oversold']}, "
                  f"Overbought={bot_config['rsi_config']['overbought']}")
            print(f"   Capital: ₹{bot_config.get('current_capital', 5000)}")
            
            # Create bot instance
            bot = AggressiveGrowthBot(bot_id=bot_id, bot_config=bot_config)
            
            # Update bot manager's status
            self.bot_manager.update_bot_status(bot_id, {
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'capital': bot_config.get('current_capital', 5000)
            })
            
            # Store instance for shared access
            self.bot_instances[bot_id] = bot
            
            # Start bot's run loop in a separate thread
            bot_thread = threading.Thread(target=bot.run, daemon=True, name=f"Bot-{bot_id}")
            bot_thread.start()
            
            print(f"✅ Bot {bot_id} started successfully")
            return bot_thread
            
        except Exception as e:
            print(f"❌ Failed to launch bot {bot_id}: {e}")
            self.bot_manager.update_bot_status(bot_id, {
                'status': 'failed',
                'error': str(e)
            })
            return None
    
    def launch_all_bots(self):
        """Launch all enabled bots"""
        enabled_bots = self.bot_manager.get_enabled_bots()
        
        if not enabled_bots:
            print("⚠️ No enabled bots found in configuration")
            return
        
        print(f"\n{'='*60}")
        print(f"🤖 DELTA BOT MULTI-INSTANCE LAUNCHER")
        print(f"{'='*60}")
        print(f"📋 Found {len(enabled_bots)} enabled bot(s)")
        
        for bot_config in enabled_bots:
            self.launch_bot_instance(bot_config['id'], bot_config)
            time.sleep(0.5)  # Stagger launches slightly
        
        self.is_running = True
        print(f"\n✅ All bots launched. Flask server starting on http://localhost:5005")
    
    def stop_all_bots(self):
        """Stop all running bots"""
        print("\n⛔ Stopping all bots...")
        for bot_id, bot in self.bot_instances.items():
            try:
                if hasattr(bot, 'stop'):
                    bot.stop()
                self.bot_manager.update_bot_status(bot_id, {'status': 'stopped'})
            except Exception as e:
                print(f"⚠️ Error stopping bot {bot_id}: {e}")
        
        self.is_running = False
        print("✅ All bots stopped")
    
    def monitor_bots(self):
        """Monitor bot health periodically"""
        while self.is_running:
            try:
                for bot_id, bot in self.bot_instances.items():
                    # Check heartbeat
                    if hasattr(bot, '_bot_heartbeat_ts'):
                        time_since_heartbeat = time.time() - bot._bot_heartbeat_ts
                        if time_since_heartbeat > 180:
                            self.bot_manager.update_bot_status(bot_id, {
                                'health': 'hung',
                                'time_since_heartbeat': time_since_heartbeat
                            })
                        else:
                            self.bot_manager.update_bot_status(bot_id, {
                                'health': 'healthy',
                                'time_since_heartbeat': time_since_heartbeat
                            })
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")


def run_flask_server(bot_launcher: BotLauncher):
    """
    Run the Flask server in the main thread
    The Flask app has access to bot_launcher via global
    """
    global _bot_launcher
    
    # Register launcher's globals with Flask app
    import apps.bot.bot_pro as bot_pro
    bot_pro.bot_manager_instance = bot_launcher.bot_manager
    bot_pro.bot_instances_registry = bot_launcher.bot_instances
    
    _bot_launcher = bot_launcher
    
    print("\n🌐 Starting Flask Server on port 5005...")
    app.run(host='0.0.0.0', port=5005, debug=False, use_reloader=False)


def main():
    """Main entry point"""
    # Load bot configuration
    config_path = os.path.join(PROJECT_ROOT, "data", "bots_config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure data/bots_config.json exists")
        sys.exit(1)
    
    # Create launcher
    launcher = BotLauncher(config_path)
    
    # Launch all enabled bots
    launcher.launch_all_bots()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=launcher.monitor_bots, daemon=True)
    monitor_thread.start()
    
    # Run Flask server in main thread
    try:
        run_flask_server(launcher)
    except KeyboardInterrupt:
        print("\n\n⛔ Received interrupt signal")
        launcher.stop_all_bots()
        sys.exit(0)


if __name__ == "__main__":
    main()
