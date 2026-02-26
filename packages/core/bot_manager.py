"""
BotManager: Orchestrates multiple bot instances with independent configurations
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class BotManager:
    """
    Manages multiple bot instances, loading configurations, spawning processes,
    and tracking bot lifecycle and status.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize BotManager
        
        Args:
            config_path: Path to bots_config.json file
        """
        self.config_path = config_path
        self.bots_config: Dict[str, Any] = {}
        self.active_bots: Dict[str, Any] = {}  # Running bot processes
        self.bot_processes: Dict[str, Any] = {}  # Process handles
        
        self.load_bots_config()
    
    def load_bots_config(self) -> Dict[str, Any]:
        """Load bots configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.bots_config = json.load(f)
        
        return self.bots_config
    
    def save_bots_config(self):
        """Save bots configuration to JSON file"""
        self.bots_config['last_updated'] = datetime.now().isoformat()
        
        with open(self.config_path, 'w') as f:
            json.dump(self.bots_config, f, indent=2)
    
    def get_all_bots(self) -> List[Dict[str, Any]]:
        """Get list of all configured bots"""
        return self.bots_config.get('bots', [])
    
    def get_bot_config(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific bot"""
        bots = self.get_all_bots()
        for bot in bots:
            if bot['id'] == bot_id:
                return bot
        return None
    
    def get_enabled_bots(self) -> List[Dict[str, Any]]:
        """Get list of enabled bots"""
        return [bot for bot in self.get_all_bots() if bot.get('enabled', False)]
    
    def set_active_bot(self, bot_id: str):
        """Set the currently active bot"""
        self.bots_config['active_bot_id'] = bot_id
        self.save_bots_config()
    
    def get_active_bot_id(self) -> str:
        """Get the currently active bot ID"""
        return self.bots_config.get('active_bot_id', 'bot_1')
    
    def add_bot(self, bot_config: Dict[str, Any]) -> str:
        """
        Add a new bot configuration
        
        Args:
            bot_config: Bot configuration dict with id, name, starting_capital, etc.
            
        Returns:
            bot_id of the created bot
        """
        bot_id = bot_config.get('id')
        if not bot_id:
            raise ValueError("bot_config must include 'id' field")
        
        # Check if bot already exists
        if self.get_bot_config(bot_id):
            raise ValueError(f"Bot with id '{bot_id}' already exists")
        
        # Create bot with defaults
        new_bot = {
            'id': bot_id,
            'name': bot_config.get('name', f'Bot {bot_id}'),
            'enabled': bot_config.get('enabled', True),
            'starting_capital': bot_config.get('starting_capital', 5000),
            'current_capital': bot_config.get('current_capital', bot_config.get('starting_capital', 5000)),
            'strategy_mix': bot_config.get('strategy_mix', {'sniper': 0.7, 'ema_cross': 0.3}),
            'rsi_config': bot_config.get('rsi_config', {'period': 14, 'oversold': 30, 'overbought': 70}),
            'macd_config': bot_config.get('macd_config', {'fast': 12, 'slow': 26, 'signal': 9}),
            'notes': bot_config.get('notes', '')
        }
        
        self.bots_config['bots'].append(new_bot)
        self.save_bots_config()
        
        return bot_id
    
    def update_bot_capital(self, bot_id: str, new_capital: float):
        """Update the current capital for a bot"""
        bot = self.get_bot_config(bot_id)
        if not bot:
            raise ValueError(f"Bot '{bot_id}' not found")
        
        # Find and update bot in config
        for b in self.bots_config['bots']:
            if b['id'] == bot_id:
                b['current_capital'] = new_capital
                break
        
        self.save_bots_config()
    
    def reset_bot_capital(self, bot_id: str):
        """Reset bot capital to starting_capital"""
        bot = self.get_bot_config(bot_id)
        if not bot:
            raise ValueError(f"Bot '{bot_id}' not found")
        
        starting_capital = bot.get('starting_capital', 5000)
        self.update_bot_capital(bot_id, starting_capital)
    
    def update_bot_config(self, bot_id: str, updates: Dict[str, Any]):
        """
        Update configuration for a bot
        
        Args:
            bot_id: Bot ID
            updates: Dict of fields to update (rsi_config, macd_config, etc.)
        """
        bot = self.get_bot_config(bot_id)
        if not bot:
            raise ValueError(f"Bot '{bot_id}' not found")
        
        # Update bot in config
        for b in self.bots_config['bots']:
            if b['id'] == bot_id:
                b.update(updates)
                break
        
        self.save_bots_config()
    
    def enable_bot(self, bot_id: str):
        """Enable a bot"""
        self.update_bot_config(bot_id, {'enabled': True})
    
    def disable_bot(self, bot_id: str):
        """Disable a bot"""
        self.update_bot_config(bot_id, {'enabled': False})
    
    def remove_bot(self, bot_id: str):
        """Remove a bot configuration"""
        bot = self.get_bot_config(bot_id)
        if not bot:
            raise ValueError(f"Bot '{bot_id}' not found")
        
        # Remove from config
        self.bots_config['bots'] = [b for b in self.bots_config['bots'] if b['id'] != bot_id]
        self.save_bots_config()
    
    def get_bot_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get status of a bot (from active_bots tracking)
        
        Returns:
            Dict with status, last_heartbeat, capital, etc.
        """
        return self.active_bots.get(bot_id, {})
    
    def update_bot_status(self, bot_id: str, status: Dict[str, Any]):
        """Update bot status"""
        self.active_bots[bot_id] = {
            **self.active_bots.get(bot_id, {}),
            **status,
            'last_update': datetime.now().isoformat()
        }
    
    def list_all_bots_with_status(self) -> List[Dict[str, Any]]:
        """Get all bots with their current status"""
        result = []
        for bot_config in self.get_all_bots():
            bot_id = bot_config['id']
            status = self.get_bot_status(bot_id)
            result.append({
                **bot_config,
                'status': status.get('status', 'stopped'),
                'last_heartbeat': status.get('last_heartbeat'),
                'health': status.get('health', 'unknown')
            })
        return result
