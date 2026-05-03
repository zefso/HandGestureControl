"""
GestureActionExecutor — виконує дії жестів з підтримкою профілів.

Профілі перемикаються прямо під час роботи (без перезапуску):
    executor.set_profile("media")

Валідація при завантаженні — якщо ключ невалідний, попереджає одразу,
а не падає в рантаймі при першому жесті.
"""

import pyautogui
import webbrowser
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Всі валідні ключі pyautogui
VALID_KEYS: set[str] = set(pyautogui.KEYBOARD_KEYS)


class ProfileNotFoundError(Exception):
    pass


class GestureActionExecutor:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._profiles: dict = {}
        self._active_profile: str = "default"
        self._reload()

    # ------------------------------------------------------------------
    # Завантаження / перезавантаження
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        """Читає gestures.json і валідує всі профілі."""
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"[Executor] Не вдалося прочитати конфіг: {e}")
            return

        self._profiles = cfg.get("profiles", {})
        requested = cfg.get("active_profile", "default")

        if requested in self._profiles:
            self._active_profile = requested
        else:
            logger.warning(f"[Executor] Профіль '{requested}' не знайдено, використовується 'default'.")
            self._active_profile = "default"

        self._validate_all()
        logger.info(f"[Executor] Завантажено профілів: {list(self._profiles.keys())}")
        logger.info(f"[Executor] Активний профіль: '{self._active_profile}'")

    def reload(self) -> None:
        """Публічний метод — перечитати файл конфігу без перезапуску."""
        self._reload()
        print(f"[Executor] Конфіг перезавантажено. Профіль: '{self._active_profile}'")

    # ------------------------------------------------------------------
    # Профілі
    # ------------------------------------------------------------------

    def set_profile(self, name: str) -> bool:
        """Перемикає активний профіль. Повертає True якщо успішно."""
        if name not in self._profiles:
            logger.warning(f"[Executor] Профіль '{name}' не існує. Доступні: {self.available_profiles}")
            return False
        self._active_profile = name
        print(f"[Executor] Профіль змінено → '{name}'")
        return True

    def next_profile(self) -> str:
        """Перемикає на наступний профіль по колу. Зручно повісити на жест."""
        names = self.available_profiles
        idx = names.index(self._active_profile) if self._active_profile in names else 0
        next_name = names[(idx + 1) % len(names)]
        self.set_profile(next_name)
        return next_name

    @property
    def active_profile(self) -> str:
        return self._active_profile

    @property
    def available_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    @property
    def active_profile_description(self) -> str:
        return self._profiles.get(self._active_profile, {}).get("description", self._active_profile)

    # ------------------------------------------------------------------
    # Виконання дії
    # ------------------------------------------------------------------

    def execute(self, gesture: str) -> str:
        """
        Виконує дію для жесту з поточного профілю.
        Повертає рядок-опис для відображення в UI.
        """
        actions = self._profiles.get(self._active_profile, {}).get("actions", {})
        action = actions.get(gesture)

        if not action:
            return gesture  # нема конфігу — просто показуємо назву

        action_type = action.get("type")
        description = action.get("description", gesture)

        try:
            if action_type == "hotkey":
                keys = action.get("keys", [])
                if keys:
                    pyautogui.hotkey(*keys)

            elif action_type == "key":
                key = action.get("key", "")
                if key:
                    pyautogui.press(key)

            elif action_type == "url":
                url = action.get("url", "")
                if url:
                    webbrowser.open(url)

            elif action_type == "command":
                cmd = action.get("cmd", "")
                if cmd:
                    os.system(cmd)

            else:
                logger.warning(f"[Executor] Невідомий тип дії: '{action_type}'")
                return gesture

        except Exception as e:
            logger.error(f"[Executor] Помилка виконання '{gesture}': {e}")
            return f"ERR: {gesture}"

        return description

    # ------------------------------------------------------------------
    # Валідація
    # ------------------------------------------------------------------

    def _validate_all(self) -> None:
        """Перевіряє всі профілі при старті. Логує попередження — не падає."""
        for profile_name, profile in self._profiles.items():
            for gesture, action in profile.get("actions", {}).items():
                self._validate_action(profile_name, gesture, action)

    def _validate_action(self, profile: str, gesture: str, action: dict) -> None:
        action_type = action.get("type")

        if action_type == "hotkey":
            for key in action.get("keys", []):
                if key not in VALID_KEYS:
                    logger.warning(
                        f"[Executor] [{profile}] '{gesture}': невалідний ключ '{key}'. "
                        f"Перевір список pyautogui.KEYBOARD_KEYS"
                    )

        elif action_type == "key":
            key = action.get("key", "")
            if key and key not in VALID_KEYS:
                logger.warning(
                    f"[Executor] [{profile}] '{gesture}': невалідний ключ '{key}'."
                )

        elif action_type == "url":
            url = action.get("url", "")
            if not url.startswith(("http://", "https://")):
                logger.warning(
                    f"[Executor] [{profile}] '{gesture}': URL '{url}' не починається з http(s)://."
                )

        elif action_type == "command":
            pass  # команди важко валідувати наперед

        else:
            logger.warning(
                f"[Executor] [{profile}] '{gesture}': невідомий тип '{action_type}'. "
                f"Очікується: hotkey | key | url | command"
            )

    def summary(self) -> str:
        """Рядок для дебагу — поточний профіль і всі дії."""
        actions = self._profiles.get(self._active_profile, {}).get("actions", {})
        lines = [f"Profile: '{self._active_profile}' — {self.active_profile_description}"]
        for gesture, action in actions.items():
            desc = action.get("description", "?")
            lines.append(f"  {gesture:<14} → {desc}")
        return "\n".join(lines)