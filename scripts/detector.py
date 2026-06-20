"""
============================================================
  IDS Project — Network Intrusion Detector
  
  Two modes:
  1. LIVE — captures packets in real time
  2. FILE — reads a Wireshark .pcap file

HOW TO RUN:
  # Live mode (requires sudo):
  sudo python scripts/detector.py --mode live --interface eth0

  # File mode:
  python scripts/detector.py --mode file --input capture.pcap
============================================================
"""

import argparse
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scapy.all import sniff, rdpcap, IP, TCP, UDP, ICMP

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

THRESHOLD   = 0.1   # classification threshold
TIME_WINDOW = 2     # seconds for time-based features
HOST_WINDOW = 100   # connections for host-based features

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_tuned_threshold01.pkl"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH   = LOG_DIR / "detections.log"

# ─────────────────────────────────────────────────────────
# PORT → SERVICE MAPPING
# ─────────────────────────────────────────────────────────

PORT_TO_SERVICE = {
    7: "echo", 
    9: "discard", 
    11: "systat", 
    13: "daytime",
    15: "netstat", 
    19: "supdup", 
    20: "ftp_data", 
    21: "ftp",
    22: "ssh", 
    23: "telnet", 
    25: "smtp", 
    37: "time",
    43: "whois", 
    53: "domain", 
    69: "tftp_u", 
    70: "gopher",
    79: "finger", 
    80: "http", 
    101: "hostnames", 
    102: "iso_tsap",
    105: "csnet_ns", 
    107: "remote_job", 
    109: "pop_2", 
    110: "pop_3",
    111: "sunrpc", 
    113: "auth", 
    117: "uucp_path", 
    119: "nntp",
    123: "ntp_u", 
    137: "netbios_ns", 
    138: "netbios_dgm",
    139: "netbios_ssn", 
    143: "imap4", 
    177: "X11", 
    179: "bgp",
    194: "IRC", 
    210: "Z39_50", 
    389: "ldap", 
    443: "http_443",
    512: "exec", 
    513: "login", 
    514: "shell", 
    515: "printer",
    517: "name", 
    530: "courier", 
    531: "IRC", 
    540: "uucp",
    543: "klogin", 
    544: "kshell", 
    601: "sql_net",
    8001: "http_8001",
}

# ICMP type → service
ICMP_TO_SERVICE = {
    0: "ecr_i",    # echo reply
    3: "urp_i",    # destination unreachable
    5: "red_i",    # redirect
    8: "eco_i",    # echo request
    11: "tim_i",   # time exceeded
}

# ─────────────────────────────────────────────────────────
# ALL FEATURE COLUMNS (must match training exactly)
# ─────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    
    "duration", "logged_in", "count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp",
    "service_IRC", "service_X11", "service_Z39_50", "service_aol",
    "service_auth", "service_bgp", "service_courier", "service_csnet_ns",
    "service_ctf", "service_daytime", "service_discard", "service_domain",
    "service_domain_u", "service_echo", "service_eco_i", "service_ecr_i",
    "service_efs", "service_exec", "service_finger", "service_ftp",
    "service_ftp_data", "service_gopher", "service_harvest",
    "service_hostnames", "service_http", "service_http_2784",
    "service_http_443", "service_http_8001", "service_imap4",
    "service_iso_tsap", "service_klogin", "service_kshell",
    "service_ldap", "service_link", "service_login", "service_mtp",
    "service_name", "service_netbios_dgm", "service_netbios_ns",
    "service_netbios_ssn", "service_netstat", "service_nnsp",
    "service_nntp", "service_ntp_u", "service_other", "service_pm_dump",
    "service_pop_2", "service_pop_3", "service_printer", "service_private",
    "service_red_i", "service_remote_job", "service_rje", "service_shell",
    "service_smtp", "service_sql_net", "service_ssh", "service_sunrpc",
    "service_supdup", "service_systat", "service_telnet", "service_tftp_u",
    "service_tim_i", "service_time", "service_urh_i", "service_urp_i",
    "service_uucp", "service_uucp_path", "service_vmnet", "service_whois",
    "flag_OTH", "flag_REJ", "flag_RSTO", "flag_RSTOS0", "flag_RSTR",
    "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH"
]


def is_valid_connection(conn: dict) -> bool:
    """
    Filter out multicast, broadcast and loopback traffic.
    """
    dst_ip = conn["dst_ip"]

    if dst_ip.startswith("224.") or dst_ip.startswith("239."):
        return False  # multicast
    if dst_ip.endswith(".255"):
        return False  # broadcast
    if dst_ip.startswith("127."):
        return False  # loopback
    if dst_ip.startswith("169.254."):
        return False  # link local

    return True

# ─────────────────────────────────────────────────────────
# CONNECTION TRACKER
# ─────────────────────────────────────────────────────────

class ConnectionTracker:
    """
    Tracks network connections and maintains sliding windows
    needed to calculate NSL-KDD traffic features.
    """

    def __init__(self):
        # Active connections being built
        # key: (src_ip, dst_ip, src_port, dst_port, protocol)
        self.active = {}

        # Completed connections history for time window (2 seconds)
        self.time_window = []

        # Completed connections history for host window (100 connections)
        self.host_window = []

    # ── Packet ingestion ──────────────────────────────────

    def process_packet(self, packet) -> dict | None:
        """
        Process a single packet.
        Returns a completed connection dict when connection closes.
        Returns None if connection is still open.
        """

        # Only process IP packets
        if not packet.haslayer(IP):
            return None

        ip = packet[IP]
        now = float(packet.time)

        # ── Extract basic packet info ─────────────────────
        src_ip    = ip.src
        dst_ip    = ip.dst
        src_bytes = len(packet)

        # ── Determine protocol and ports ──────────────────
        if packet.haslayer(TCP):
            proto    = "tcp"
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            flags    = packet[TCP].flags

        elif packet.haslayer(UDP):
            proto    = "udp"
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            flags    = None

        elif packet.haslayer(ICMP):
            proto    = "icmp"
            src_port = 0
            dst_port = packet[ICMP].type
            flags    = None

        else:
            return None

        # ── Connection key ────────────────────────────────
        conn_key = (src_ip, dst_ip, src_port, dst_port, proto)

        # ── Create new connection if not seen before ──────
        if conn_key not in self.active:
            self.active[conn_key] = {
                "src_ip":      src_ip,
                "dst_ip":      dst_ip,
                "src_port":    src_port,
                "dst_port":    dst_port,
                "protocol":    proto,
                "start_time":  now,
                "last_time":   now,
                "src_bytes":   0,
                "dst_bytes":   0,
                "had_syn":     False,
                "had_syn_ack": False,
                "had_fin":     False,
                "had_rst":     False,
                "completed":   False,
                "flags_seen":  set(),
            }

        conn = self.active[conn_key]
        conn["last_time"] = now
        conn["src_bytes"] += src_bytes
        # ── Track TCP flags ───────────────────────────────
        if proto == "tcp" and flags is not None:
            if flags & 0x02:   # SYN
                conn["had_syn"] = True
            if flags & 0x12:   # SYN + ACK
                conn["had_syn_ack"] = True
            if flags & 0x01:   # FIN
                conn["had_fin"] = True
            if flags & 0x04:   # RST
                conn["had_rst"] = True
            conn["flags_seen"].add(int(flags))

        # ── Check if connection is complete ───────────────
        completed = False

        if proto == "tcp":
            # Connection complete if FIN or RST seen
            if conn["had_fin"] or conn["had_rst"]:
                completed = True

        elif proto == "udp" or proto == "icmp":
            # UDP/ICMP: complete after 5 seconds of inactivity
            if now - conn["start_time"] > 5:
                completed = True

        # ── Return completed connection ───────────────────
        if completed:
            conn["completed"] = True
            conn["duration"]  = conn["last_time"] - conn["start_time"]
            finished = self.active.pop(conn_key)
            self._add_to_history(finished)
            return finished

        return None

    # ── History management ────────────────────────────────

    def _add_to_history(self, conn: dict) -> None:
        """Add completed connection to both history windows."""
        now = conn["last_time"]

        # Add to time window
        self.time_window.append(conn)

        # Remove connections older than TIME_WINDOW seconds
        self.time_window = [
            c for c in self.time_window
            if now - c["last_time"] <= TIME_WINDOW
        ]

        # Add to host window (keep last HOST_WINDOW connections)
        self.host_window.append(conn)
        if len(self.host_window) > HOST_WINDOW:
            self.host_window.pop(0)

    def flush_stale(self, timeout: float = 60) -> list[dict]:
        """
        Force-complete connections that have been open too long.
        Called periodically to avoid memory buildup.
        """
        now = time.time()
        stale = []
        stale_keys = []

        for key, conn in self.active.items():
            if now - conn["last_time"] > timeout:
                conn["completed"] = False
                conn["duration"]  = conn["last_time"] - conn["start_time"]
                stale.append(conn)
                stale_keys.append(key)

        for key in stale_keys:
            finished = self.active.pop(key)
            self._add_to_history(finished)

        return stale


# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────

def get_service(protocol: str, dst_port: int) -> str:
    """
    Map protocol and destination port to NSL-KDD service name.
    """
    if protocol == "icmp":
        return ICMP_TO_SERVICE.get(dst_port, "other")
    elif protocol == "udp":
        if dst_port == 53:
            return "domain_u"
        return PORT_TO_SERVICE.get(dst_port, "other")
    else:
        return PORT_TO_SERVICE.get(dst_port, "private")


def get_flag(conn: dict) -> str:
    """
    Map TCP connection state to NSL-KDD flag value.
    """
    if conn["protocol"] != "tcp":
        return "SF"

    had_syn     = conn["had_syn"]
    had_syn_ack = conn["had_syn_ack"]
    had_fin     = conn["had_fin"]
    had_rst     = conn["had_rst"]

    if had_fin and not had_rst:
        return "SF"        # normal complete connection
    elif had_rst and had_syn_ack:
        return "RSTR"      # reset by responder
    elif had_rst and not had_syn_ack:
        return "RSTO"      # reset by originator
    elif had_syn and not had_syn_ack and not had_rst:
        return "S0"        # SYN sent — no response (SYN flood)
    elif had_syn and conn["had_fin"]:
        return "SH"        # SYN + FIN unusual
    elif had_syn_ack and not had_fin:
        return "S1"        # SYN-ACK but not finished
    else:
        return "OTH"       # anything else


def extract_features(conn: dict, tracker: ConnectionTracker) -> pd.DataFrame:
    """
    Extract all 97 features from a completed connection.
    Returns a single row DataFrame ready for the model.
    """

    # ── Start with all zeros ──────────────────────────────
    features = {col: 0 for col in FEATURE_COLUMNS}

    # ── Group 1 — Per connection features ────────────────
    features["duration"] = round(conn.get("duration", 0), 4)
    features["logged_in"] = 1 if conn["had_syn_ack"] else 0

    # ── Protocol One Hot Encoding ─────────────────────────
    proto = conn["protocol"]
    if proto == "tcp":
        features["protocol_type_tcp"]  = 1
    elif proto == "udp":
        features["protocol_type_udp"]  = 1
    elif proto == "icmp":
        features["protocol_type_icmp"] = 1

    # ── Service One Hot Encoding ──────────────────────────
    service = get_service(proto, conn["dst_port"])
    service_col = f"service_{service}"
    if service_col in features:
        features[service_col] = 1

    # ── Flag One Hot Encoding ─────────────────────────────
    flag = get_flag(conn)
    flag_col = f"flag_{flag}"
    if flag_col in features:
        features[flag_col] = 1

    # ── Group 3 — Time window features (last 2 seconds) ───
    now          = conn["last_time"]
    dst_ip       = conn["dst_ip"]
    dst_port     = conn["dst_port"]
    src_port     = conn["src_port"]
    current_svc  = service

    # Filter time window to last 2 seconds
    recent = [
        c for c in tracker.time_window
        if now - c["last_time"] <= TIME_WINDOW
    ]

    # count — connections to same destination host in last 2 seconds
    same_host = [c for c in recent if c["dst_ip"] == dst_ip]
    count = len(same_host)
    features["count"] = count

    if count > 0:
        # serror_rate — % with SYN errors (S0 flag)
        syn_errors = sum(
            1 for c in same_host
            if get_flag(c) == "S0"
        )
        features["serror_rate"] = round(syn_errors / count, 4)

        # rerror_rate — % rejected (REJ flag)
        rej_errors = sum(
            1 for c in same_host
            if get_flag(c) == "REJ"
        )
        features["rerror_rate"] = round(rej_errors / count, 4)

        # same_srv_rate — % to same service
        same_svc = sum(
            1 for c in same_host
            if get_service(c["protocol"], c["dst_port"]) == current_svc
        )
        features["same_srv_rate"] = round(same_svc / count, 4)

        # diff_srv_rate — % to different services
        features["diff_srv_rate"] = round(
            1 - features["same_srv_rate"], 4
        )

    # srv_diff_host_rate — % of same-service connections
    # going to different hosts
    same_svc_conns = [
        c for c in recent
        if get_service(c["protocol"], c["dst_port"]) == current_svc
    ]
    srv_count = len(same_svc_conns)

    if srv_count > 0:
        diff_hosts = sum(
            1 for c in same_svc_conns
            if c["dst_ip"] != dst_ip
        )
        features["srv_diff_host_rate"] = round(
            diff_hosts / srv_count, 4
        )

    # ── Group 4 — Host window features (last 100 conns) ───
    host_conns = tracker.host_window

    # dst_host_count — connections to same destination host
    dst_host_conns = [
        c for c in host_conns if c["dst_ip"] == dst_ip
    ]
    dst_host_count = len(dst_host_conns)
    features["dst_host_count"] = dst_host_count

    if dst_host_count > 0:

        # dst_host_srv_count — same service on same host
        dst_host_svc_conns = [
            c for c in dst_host_conns
            if get_service(c["protocol"], c["dst_port"]) == current_svc
        ]
        dst_host_srv_count = len(dst_host_svc_conns)
        features["dst_host_srv_count"] = dst_host_srv_count

        # dst_host_same_srv_rate
        features["dst_host_same_srv_rate"] = round(
            dst_host_srv_count / dst_host_count, 4
        )

        # dst_host_diff_srv_rate
        features["dst_host_diff_srv_rate"] = round(
            1 - features["dst_host_same_srv_rate"], 4
        )

        # dst_host_same_src_port_rate
        same_src_port = sum(
            1 for c in dst_host_conns
            if c["src_port"] == src_port
        )
        features["dst_host_same_src_port_rate"] = round(
            same_src_port / dst_host_count, 4
        )

        # dst_host_srv_diff_host_rate
        if dst_host_srv_count > 0:
            svc_diff_hosts = sum(
                1 for c in dst_host_svc_conns
                if c["dst_ip"] != dst_ip
            )
            features["dst_host_srv_diff_host_rate"] = round(
                svc_diff_hosts / dst_host_srv_count, 4
            )

    # ── Return as DataFrame ───────────────────────────────
    return pd.DataFrame([features])[FEATURE_COLUMNS]




 # ─────────────────────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────────────────────


#function to load the model from disk using joblib
def load_model():
    """
    Load the saved Random Forest model from disk.
    """
    print(f"  Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"  ✅ Model loaded successfully")
    return model

def detect(conn: dict, features: pd.DataFrame, model) -> tuple[bool, float]:
    """
    Run the model on extracted features.
    Uses higher threshold for HTTPS traffic to reduce false positives
    from modern web browsing.
    """
    # Get probability of being an attack (class 1)
    probability = model.predict_proba(features)[0][1]

    # Modern HTTPS traffic looks suspicious to 1998 trained model
    # Use higher threshold for port 443 to reduce false positives
    if conn["dst_port"] == 443 or conn["src_port"] == 443:
        is_attack = probability >= 0.55
    else:
        is_attack = probability >= THRESHOLD

    return is_attack, round(probability * 100, 2)



# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

def log_detection(conn: dict, probability: float) -> None:
    """
    Print attack detection to terminal and save to log file.
    """

    # Get service and flag for display
    service = get_service(conn["protocol"], conn["dst_port"])
    flag    = get_flag(conn)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Terminal output ───────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ⚠️  ATTACK DETECTED")
    print(f"{'='*55}")
    print(f"  Timestamp  : {timestamp}")
    print(f"  Source     : {conn['src_ip']}:{conn['src_port']}")
    print(f"  Target     : {conn['dst_ip']}:{conn['dst_port']}")
    print(f"  Protocol   : {conn['protocol'].upper()}")
    print(f"  Service    : {service}")
    print(f"  Flag       : {flag}")
    print(f"  Duration   : {conn.get('duration', 0):.4f}s")
    print(f"  Confidence : {probability:.1f}%")
    print(f"{'='*55}\n")

    # ── Log file output ───────────────────────────────────
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp} | ATTACK | "
                f"src={conn['src_ip']}:{conn['src_port']} | "
                f"dst={conn['dst_ip']}:{conn['dst_port']} | "
                f"proto={conn['protocol'].upper()} | "
                f"service={service} | "
                f"flag={flag} | "
                f"confidence={probability:.1f}%\n")


def log_normal(conn: dict) -> None:
    """
    Print normal connection to terminal (optional verbose mode).
    """
    service   = get_service(conn["protocol"], conn["dst_port"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  ✅ Normal  | {timestamp} | "
          f"{conn['src_ip']} → {conn['dst_ip']}:{conn['dst_port']} | "
          f"{service}")
    



# ─────────────────────────────────────────────────────────
# PACKET HANDLER
# ─────────────────────────────────────────────────────────

def handle_packet(packet, tracker: ConnectionTracker, 
                  model, verbose: bool = False) -> None:
    
    completed_conn = tracker.process_packet(packet)
    if completed_conn is None:
        return

    # ── Add this check ────────────────────────────────────
    if not is_valid_connection(completed_conn):
        return
    # ─────────────────────────────────────────────────────

    try:
        features = extract_features(completed_conn, tracker)
    except Exception as e:
        return

    is_attack, probability = detect(completed_conn, features, model)

    if is_attack:
        if probability >= 30.0:
            log_detection(completed_conn, probability)
    elif verbose:
        log_normal(completed_conn)
  


# ─────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────

def main():
    """ 
    Entry point — interactively asks user for mode selection.
    """

    # ── Startup banner ────────────────────────────────────
    print("\n" + "="*55)
    print("  IDS — Network Intrusion Detection System")
    print("="*55)
    print(f"  Threshold : {THRESHOLD}")
    print(f"  Log file  : {LOG_PATH}")
    print("="*55 + "\n")

    # ── Load model ────────────────────────────────────────
    model = load_model()

    # ── Ask user for mode ─────────────────────────────────
    print("\n  Select mode:")
    print("  1 → Live capture (listens to real network traffic)")
    print("  2 → File mode    (reads a Wireshark .pcap file)")

    while True:
        choice = input("\n  Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            break
        print("  ❌ Invalid choice — please enter 1 or 2")

    # ── Create connection tracker ─────────────────────────
    tracker = ConnectionTracker()

    # ── Mode 1 — Live capture ─────────────────────────────
    if choice == "1":

        print("\n  Available network interfaces:")
        from scapy.all import get_if_list
        interfaces = get_if_list()
        for i, iface in enumerate(interfaces):
            print(f"    {i + 1} → {iface}")

        interface = input(
            "\n  Enter interface name (e.g. eth0): "
        ).strip()

        verbose = input(
            "  Show normal connections too? (y/n): "
        ).strip().lower() == "y"

        print(f"\n  Listening on: {interface}")
        print(f"  Press Ctrl+C to stop\n")

        try:
            sniff(
                iface=interface,
                prn=lambda pkt: handle_packet(
                    pkt, tracker, model, verbose
                ),
                store=False
            )
        except KeyboardInterrupt:
            print("\n\n  Capture stopped.")
            print(f"  Detections saved to: {LOG_PATH}")

    # ── Mode 2 — File capture ─────────────────────────────
    
    elif choice == "2":

        pcap_input = input(
            "\n  Enter path to .pcap file: "
        ).strip()

        pcap_path = Path(pcap_input)

        if not pcap_path.exists():
            print(f"  ❌ File not found: {pcap_path}")
            return

        verbose = input(
            "  Show normal connections too? (y/n): "
        ).strip().lower() == "y"

        print(f"\n  Reading: {pcap_path}")
        packets = rdpcap(str(pcap_path))
        print(f"  Packets loaded: {len(packets):,}")
        print(f"  Processing... (Press Ctrl+C to stop early)\n")

        try:
            for i, packet in enumerate(packets):
                handle_packet(packet, tracker, model, verbose)

                # Progress update every 1000 packets
                if i % 1000 == 0 and i > 0:
                    print(f"  Processed {i:,} / {len(packets):,} packets...")

                # Flush stale connections every 1000 packets
                if i % 1000 == 0:
                    stale = tracker.flush_stale()
                    for conn in stale:
                        try:
                            features  = extract_features(conn, tracker)
                            is_attack, probability = detect(
                                conn, features, model
                            )
                            if is_attack:
                                log_detection(conn, probability)
                        except Exception:
                            pass

        except KeyboardInterrupt:
            print(f"\n\n  Processing stopped by user at packet {i:,}")

        # Final flush regardless of how we stopped
        print("\n  Processing remaining connections...")
        stale = tracker.flush_stale(timeout=0)
        for conn in stale:
            try:
                features  = extract_features(conn, tracker)
                is_attack, probability = detect(conn, features, model)
                if is_attack:
                    log_detection(conn, probability)
            except Exception:
                pass

        print(f"\n  ✅ Processing complete")
        print(f"  Detections saved to: {LOG_PATH}")

if __name__ == "__main__":
    main()