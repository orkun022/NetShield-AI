import os
import csv
import random
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'pop3', 'imap', 'https', 'other']
FLAGS = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH']
ATTACK_TYPES = ['normal', 'neptune', 'smurf', 'back', 'ipsweep', 'portsweep',
                'satan', 'nmap', 'guess_passwd', 'buffer_overflow', 'rootkit']


def generate_normal_traffic():
    return {
        'duration': random.randint(0, 300),
        'protocol_type': random.choice(['tcp', 'tcp', 'tcp', 'udp']),
        'service': random.choice(['http', 'http', 'https', 'dns', 'smtp', 'ssh']),
        'flag': random.choice(['SF', 'SF', 'SF', 'S0']),
        'src_bytes': random.randint(100, 10000),
        'dst_bytes': random.randint(100, 50000),
        'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': random.randint(0, 3),
        'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
        'num_root': 0, 'num_file_creations': random.randint(0, 2),
        'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0,
        'count': random.randint(1, 50),
        'srv_count': random.randint(1, 50),
        'serror_rate': round(random.uniform(0, 0.1), 2),
        'srv_serror_rate': round(random.uniform(0, 0.1), 2),
        'rerror_rate': round(random.uniform(0, 0.1), 2),
        'srv_rerror_rate': round(random.uniform(0, 0.1), 2),
        'same_srv_rate': round(random.uniform(0.8, 1.0), 2),
        'diff_srv_rate': round(random.uniform(0, 0.2), 2),
        'srv_diff_host_rate': round(random.uniform(0, 0.1), 2),
        'dst_host_count': random.randint(1, 255),
        'dst_host_srv_count': random.randint(1, 255),
        'dst_host_same_srv_rate': round(random.uniform(0.5, 1.0), 2),
        'dst_host_diff_srv_rate': round(random.uniform(0, 0.3), 2),
        'dst_host_same_src_port_rate': round(random.uniform(0, 0.5), 2),
        'dst_host_srv_diff_host_rate': round(random.uniform(0, 0.1), 2),
        'dst_host_serror_rate': round(random.uniform(0, 0.1), 2),
        'dst_host_srv_serror_rate': round(random.uniform(0, 0.1), 2),
        'dst_host_rerror_rate': round(random.uniform(0, 0.1), 2),
        'dst_host_srv_rerror_rate': round(random.uniform(0, 0.1), 2),
        'label': 'normal',
    }


def generate_dos_attack():
    traffic = generate_normal_traffic()
    traffic.update({
        'duration': 0,
        'service': random.choice(['http', 'http', 'ecr_i', 'other']),
        'flag': random.choice(['SF', 'S0', 'S0', 'REJ']),
        'src_bytes': random.choice([0, random.randint(0, 100)]),
        'dst_bytes': random.randint(0, 1000),
        'count': random.randint(100, 511),
        'srv_count': random.randint(1, 10),
        'serror_rate': round(random.uniform(0.5, 1.0), 2),
        'srv_serror_rate': round(random.uniform(0.5, 1.0), 2),
        'same_srv_rate': round(random.uniform(0, 0.3), 2),
        'dst_host_serror_rate': round(random.uniform(0.5, 1.0), 2),
        'label': random.choice(['neptune', 'smurf', 'back']),
    })
    return traffic


def generate_probe_attack():
    traffic = generate_normal_traffic()
    traffic.update({
        'duration': random.randint(0, 5),
        'flag': random.choice(['S0', 'REJ', 'RSTR', 'SF']),
        'src_bytes': random.randint(0, 500),
        'dst_bytes': 0,
        'count': random.randint(1, 50),
        'srv_count': random.randint(1, 5),
        'rerror_rate': round(random.uniform(0.3, 1.0), 2),
        'dst_host_count': random.randint(200, 255),
        'dst_host_diff_srv_rate': round(random.uniform(0.3, 1.0), 2),
        'label': random.choice(['ipsweep', 'portsweep', 'satan', 'nmap']),
    })
    return traffic


def generate_r2l_attack():
    traffic = generate_normal_traffic()
    traffic.update({
        'duration': random.randint(1, 1000),
        'protocol_type': 'tcp',
        'service': random.choice(['ftp', 'telnet', 'smtp', 'imap', 'pop3']),
        'flag': 'SF',
        'num_failed_logins': random.randint(1, 5),
        'logged_in': random.choice([0, 0, 1]),
        'hot': random.randint(2, 10),
        'label': random.choice(['guess_passwd', 'guess_passwd', 'ftp_write']),
    })
    return traffic


def generate_u2r_attack():
    traffic = generate_normal_traffic()
    traffic.update({
        'duration': random.randint(1, 500),
        'protocol_type': 'tcp',
        'service': random.choice(['telnet', 'ftp', 'ssh']),
        'flag': 'SF',
        'logged_in': 1,
        'root_shell': 1,
        'su_attempted': random.choice([0, 1, 2]),
        'num_root': random.randint(1, 10),
        'num_shells': random.randint(1, 3),
        'num_file_creations': random.randint(3, 15),
        'label': random.choice(['buffer_overflow', 'rootkit']),
    })
    return traffic


def generate_and_save(n_normal=500, n_dos=200, n_probe=150, n_r2l=100, n_u2r=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    data = []
    for _ in range(n_normal):
        data.append(generate_normal_traffic())
    for _ in range(n_dos):
        data.append(generate_dos_attack())
    for _ in range(n_probe):
        data.append(generate_probe_attack())
    for _ in range(n_r2l):
        data.append(generate_r2l_attack())
    for _ in range(n_u2r):
        data.append(generate_u2r_attack())

    random.shuffle(data)

    filepath = os.path.join(PROJECT_ROOT, 'data', 'raw', 'network_traffic.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    columns = list(data[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

    print(f"[+] Dataset kaydedildi: {filepath} ({len(data)} kayit)")
    return filepath


def main():
    generate_and_save()


if __name__ == '__main__':
    main()
