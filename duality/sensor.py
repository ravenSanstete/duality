# this file invokes baidu's RESTful api to do recognition of the speech
# this part has already been done at the client ends

# [deprecated]



import pycurl
import io
import json
import base64
from uuid import getnode

"""
the general interface

@param: pcm_data: in a binary array form
"""

def recognize(pcm_data):
    token = _auth();
    return _invoke_recognition(pcm_data, token);


def get_mac_addr():
    return ":".join([hex(getnode())[2*i:2*i+2] for i in range(1,7)]);



"""
@return: the access token
"""
def _auth():
    buffer = io.BytesIO();
    api_key='R9YSu4GyH7Fc0j80S5GT1Uou';
    client_secret='02b86feade35d81aa0feb2743fc43831';
    auth_url="https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" % (api_key, client_secret);
    curl_obj=pycurl.Curl();
    curl_obj.setopt(pycurl.URL, auth_url);
    curl_obj.setopt(pycurl.TIMEOUT, 5);
    curl_obj.setopt(pycurl.WRITEDATA, buffer);
    curl_obj.perform();
    token=None;
    if(curl_obj.getinfo(pycurl.HTTP_CODE)==200):
        token=json.loads(buffer.getvalue().decode('utf-8'))['access_token'];
    curl_obj.close();
    return token;



"""
@param: pcm_data
@return: the curl instance
"""
def _invoke_recognition(pcm_data, token):
    cuid='9321748';
    url = "http://vop.baidu.com/server_api";
    buffer=io.BytesIO();
    curl_obj=pycurl.Curl();
    encoded_speech=base64.b64encode(pcm_data).decode('utf-8');

    # construct the json request string
    json_req = json.dumps({
        'format':'pcm',
        'rate':8000,
        'channel':1,
        'lan':'zh',
        'token':token,
        'cuid':get_mac_addr(),
        'len':len(pcm_data),
        'speech':encoded_speech
    })



    header=['Content-Length:%d'%(len(json_req)), 'Content-Type: application/json; charset=utf-8'];
    curl_obj.setopt(pycurl.URL, url);
    curl_obj.setopt(pycurl.HTTPHEADER, header);
    curl_obj.setopt(pycurl.POST, 1);
    curl_obj.setopt(pycurl.CONNECTTIMEOUT, 30);
    curl_obj.setopt(pycurl.TIMEOUT, 30);
    curl_obj.setopt(pycurl.POSTFIELDS, json_req);
    curl_obj.setopt(pycurl.WRITEDATA, buffer);
    curl_obj.perform();
    outcome=json.loads(buffer.getvalue().decode('utf-8'))
    curl_obj.close();

    result=None;
    if(outcome['err_no']==0):
        result=outcome['result'];
    return result;








"""
an auxiliary function, which reads binary data into a buffer from a file
@param: path, the speech file path
"""
def read_file(path):
    f=open(path, 'rb');
    bytes_data= f.read();
    f.close();
    return bytes_data;


if __name__ == '__main__':
    test_path='/Users/morino/Desktop/common/OneDrive/code/Baidu_Voice_RestApi_SampleCode/sample/test.pcm';
    pcm_data = read_file(test_path);
    print(recognize(pcm_data));
