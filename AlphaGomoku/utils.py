import time
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import smtplib


from_addr = "reposter@sina.com"
password = "secret"


def send_email_report(to_addr, content):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = Header('Gomoku AI Report', 'utf-8')
        msg['From'] = Header(from_addr)
        msg['To'] = Header(to_addr)
        msg['Reply-to'] = Header(from_addr)

        msg.attach(MIMEText(content, 'plain', 'utf-8'))

        smtp_server = "smtp.sina.com"
        server = smtplib.SMTP(smtp_server, 25)

        server.set_debuglevel(1)
        server.starttls()

        server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
    except:
        pass


def log(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        print('>> calling %s()' % func.__name__)
        result = func(*args, **kwargs)
        end = time.clock()
        print('>> %s() time = %s' % (func.__name__, str(round(end - start, 3))))
        return result

    return wrapper


def index2coordinate(index, size):
    row = index // size
    col = index % size
    return int(row), int(col)


def coordinate2index(cor, size):
    return size * cor[0] + cor[1]


def board2legalvec(board):
    vec = np.array(np.array(board) == 0, dtype=np.int)
    return vec.flatten()
