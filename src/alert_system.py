import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import platform

# Check for Windows to use winsound, else define dummy or alternative
try:
    import winsound
    system_platform = "Windows"
except ImportError:
    system_platform = "Other"

class AlertSystem:
    def __init__(self, config=None):
        self.config = config or {}
        self.email_config = self.config.get('alerts', {}).get('email_config', {})
        self.audio_enabled = self.config.get('alerts', {}).get('enable_audio', True)
        self.email_enabled = self.config.get('alerts', {}).get('enable_email', False)
        
        self.last_audio_time = 0
        self.audio_cooldown = 2.0  # Seconds between audio alerts to prevent spamming
        self.is_email_sending = False

    def play_audio(self, alert_type):
        """Plays an audio alert based on type."""
        if not self.audio_enabled:
            return

        current_time = time.time()
        if current_time - self.last_audio_time < self.audio_cooldown:
            return

        self.last_audio_time = current_time

        def _sound():
            try:
                if system_platform == "Windows":
                    if alert_type == "warning":
                        # 1000 Hz, 500 ms
                        winsound.Beep(1000, 500) 
                    elif alert_type == "critical":
                        # 2000 Hz, 300 ms (repeated rapidly could be done in loop, but here single beep)
                        winsound.Beep(2500, 800)
                else:
                    # Linux/Mac fallback (print or other libs)
                    print(f"\a[AUDIO ALERT]: {alert_type}")
            except Exception as e:
                print(f"Audio Error: {e}")

        # Run in thread to not block main loop
        threading.Thread(target=_sound, daemon=True).start()

    def send_email(self, subject, body):
        """Sends an email alert in a separate thread."""
        if not self.email_enabled or self.is_email_sending:
            return

        if not self.email_config.get('sender_email'):
            print("Email config missing. Skipping email.")
            return

        def _email_thread():
            self.is_email_sending = True
            try:
                sender_email = self.email_config['sender_email']
                sender_password = self.email_config['sender_password']
                receiver_email = self.email_config['receiver_email']
                
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = receiver_email
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()
                print(f"Alert email sent to {receiver_email}")
            except Exception as e:
                print(f"Failed to send email: {e}")
            finally:
                self.is_email_sending = False

        threading.Thread(target=_email_thread, daemon=True).start()

    def trigger(self, level, message):
        """
        Triggers alerts based on severity level.
        Level: 'warning', 'critical'
        """
        if level == "critical":
            self.play_audio("critical")
            if self.email_enabled:
                self.send_email("CRITICAL SAFETY ALERT", f"Safety System Triggered: {message}")
        elif level == "warning":
            self.play_audio("warning")
