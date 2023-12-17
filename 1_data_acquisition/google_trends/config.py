''' Configuration file for Google Trends API scraping. '''

explore_url = "https://trends.google.com/trends/api/explore"
widget_url = 'https://trends.google.com/trends/api/widgetdata/multiline/csv'

cookies = {
    'AEC': 'ARSKqsJPmyKVBKRgVpOg8x5Eo5CQWUjQj2T-5VQXZ_yk8H50lfg59sVw5Ac',
    'CONSENT': 'PENDING+370',
    'SOCS': 'CAISHAgCEhJnd3NfMjAyMjEwMTAtMF9SQzIaAmRlIAEaBgiAn8KaBg',
    'NID': '511=ckecD3zFQ7xkUJnhDS65Q2mTPAmTZxftcOG3w5jNzZpte9WkpDEppoF0o5Zf5QbTS_mzS8QXKErWk-dHkrKRVFQGLipZVq714rTPsvwSjG4bQk3Ew9INhPeUrdXXBcutqRoiBKlQHlns-9RCrYYAiV3h50HNSclVCllL20xtqdKh7CxRybfdsnCdJNshC2meTiWBIYaMk9yVARKViwV7rEyypjUCFZBTzQXIr5JRsDB6JveoEryPBHmInqpXETKJ1W7gJtgkAhm2fPu08QUHdM-6NgdLg7wuv0CS_MSPjpAWNVRXdakjm6x8aTM4EM_8Ro0BTVtUUHzWQYR_j1t2YNKK_PrLTSGUH5Qj-G3ihVG7kr6NEzQ',
    '1P_JAR': '2023-3-22-15',
    'SID': 'UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw6prfUsI8Vfl1tVXY4FfH-6w.',
    '__Secure-1PSID': 'UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw6APqdoYNi9df7mOI2z9nUgQ.',
    '__Secure-3PSID': 'UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw69XBQu_CybCuzfj489cD39Q.',
    'HSID': 'AHzY2HPdjeK64wCct',
    'SSID': 'A2JxQyMVJif-7VJJR',
    'APISID': 'xSp2fQRI1IiO5sBi/AKLP-JKjKXn66xbE0',
    'SAPISID': 'IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D',
    '__Secure-1PAPISID': 'IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D',
    '__Secure-3PAPISID': 'IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D',
    'SIDCC': 'AFvIBn-4Twm2_8uiMIgDxiniVY_ItHmDUuT6MHf0ylJ8jicqiEVaFAFDTPaTn8ExWIKpYQuz1ZXK',
    '__Secure-1PSIDCC': 'AFvIBn9nI2SrIj35gWhFMwf3Aplpm9gkH_rvUXbkePqC5Se7c6cc5uEf62ej9P2wjfF6pFlYKZs',
    '__Secure-3PSIDCC': 'AFvIBn-LsV9Iyb6Qe-1CwMXqe5mx-LZQw7wsVoTRgv2FmCmTY-OdqBCzzxHMxGorowsMbL3c0cw',
    'S': 'billing-ui-v3=ZiGzFrx4pjuLbyPhLQacqi-6H4yzf_rW:billing-ui-v3-efe=ZiGzFrx4pjuLbyPhLQacqi-6H4yzf_rW',
}

headers = {
    'Host': 'trends.google.com',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Alt-Used': 'trends.google.com',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Sec-GPC': '1',
    'Cookie': 'AEC=ARSKqsJPmyKVBKRgVpOg8x5Eo5CQWUjQj2T-5VQXZ_yk8H50lfg59sVw5Ac; CONSENT=PENDING+370; SOCS=CAISHAgCEhJnd3NfMjAyMjEwMTAtMF9SQzIaAmRlIAEaBgiAn8KaBg; NID=511=ckecD3zFQ7xkUJnhDS65Q2mTPAmTZxftcOG3w5jNzZpte9WkpDEppoF0o5Zf5QbTS_mzS8QXKErWk-dHkrKRVFQGLipZVq714rTPsvwSjG4bQk3Ew9INhPeUrdXXBcutqRoiBKlQHlns-9RCrYYAiV3h50HNSclVCllL20xtqdKh7CxRybfdsnCdJNshC2meTiWBIYaMk9yVARKViwV7rEyypjUCFZBTzQXIr5JRsDB6JveoEryPBHmInqpXETKJ1W7gJtgkAhm2fPu08QUHdM-6NgdLg7wuv0CS_MSPjpAWNVRXdakjm6x8aTM4EM_8Ro0BTVtUUHzWQYR_j1t2YNKK_PrLTSGUH5Qj-G3ihVG7kr6NEzQ; 1P_JAR=2023-3-22-15; SID=UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw6prfUsI8Vfl1tVXY4FfH-6w.; __Secure-1PSID=UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw6APqdoYNi9df7mOI2z9nUgQ.; __Secure-3PSID=UQj-jgPOHTejO_RyWPwJwKDcuvlufH3cVRdQzzaJAEmpECw69XBQu_CybCuzfj489cD39Q.; HSID=AHzY2HPdjeK64wCct; SSID=A2JxQyMVJif-7VJJR; APISID=xSp2fQRI1IiO5sBi/AKLP-JKjKXn66xbE0; SAPISID=IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D; __Secure-1PAPISID=IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D; __Secure-3PAPISID=IvMsQQtZtaePypUV/ABhatNJf5OZyivN8D; SIDCC=AFvIBn-4Twm2_8uiMIgDxiniVY_ItHmDUuT6MHf0ylJ8jicqiEVaFAFDTPaTn8ExWIKpYQuz1ZXK; __Secure-1PSIDCC=AFvIBn9nI2SrIj35gWhFMwf3Aplpm9gkH_rvUXbkePqC5Se7c6cc5uEf62ej9P2wjfF6pFlYKZs; __Secure-3PSIDCC=AFvIBn-LsV9Iyb6Qe-1CwMXqe5mx-LZQw7wsVoTRgv2FmCmTY-OdqBCzzxHMxGorowsMbL3c0cw; S=billing-ui-v3=ZiGzFrx4pjuLbyPhLQacqi-6H4yzf_rW:billing-ui-v3-efe=ZiGzFrx4pjuLbyPhLQacqi-6H4yzf_rW',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Referer': 'https://trends.google.com/trends/explore?date=2022-06-26%202023-03-22&q=bitcoin&hl=de',
}
