using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Text;
using UnityEngine;


/**
This class is used for wrapping static method for recognizing speech clip through online service

pure c# class
*/


public class Sensor {
	//feed in a byte_stream, which is of a wav file's. And output a string returned by baidu
	public static string recognize(Byte[] bytes_arr){
		Debug.Log (bytes_arr.Length);
		string token = auth ();

		return invoke_recognize(bytes_arr, token);
	}




//	public static string Base64Encode(Byte[] bytes_arr)    
//	{    
//		string go = Convert.ToBase64String(bytes_arr); 
//
//	}    


	public static string invoke_recognize(Byte[] bytes_arr, string token){
		RecogRequest req = new RecogRequest ();


		req.format="wav";
		req.rate = 8000;
		req.channel = 1;

		req.lan="en";
		req.token = token;

		Debug.Log (Net.DEVICE_ID);
		req.cuid= Net.DEVICE_ID; //will be modified later
		req.len=bytes_arr.Length;
		req.speech = Convert.ToBase64String(bytes_arr);

	

		string json_content = JsonUtility.ToJson (req);
		string url = "http://vop.baidu.com/server_api";

		WebRequest web_req = WebRequest.Create (url);
		web_req.Method="POST";

		web_req.ContentLength = json_content.Length;
		web_req.ContentType = "application/json; charset=utf-8";
		Stream req_stream = web_req.GetRequestStream ();
		byte [] byte_content = Encoding.UTF8.GetBytes (json_content);
		req_stream.Write(byte_content,0 , byte_content.Length);
		req_stream.Flush ();
		req_stream.Close ();


		Debug.Log ("write finished");
		WebResponse wr = web_req.GetResponse();
		Stream receiveStream = wr.GetResponseStream();
	
		StreamReader reader = new StreamReader(receiveStream, Encoding.UTF8);
		string content = reader.ReadToEnd ();

		receiveStream.Close ();
		reader.Close ();



		RecogResponse res_json = JsonUtility.FromJson<RecogResponse> (content);

		if (res_json.err_no == 0 && res_json.result.Length>0) {
			return res_json.result [0];
		} else {
			return res_json.err_msg;
		}
	}


	public static string auth(){
		string api_key= "R9YSu4GyH7Fc0j80S5GT1Uou";
		string client_secret="02b86feade35d81aa0feb2743fc43831";
		string auth_url = String.Format ("https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}", api_key, client_secret);

		WebRequest req = WebRequest.Create (auth_url);
		WebResponse wr = req.GetResponse();
		Stream receiveStream = wr.GetResponseStream();
		StreamReader reader = new StreamReader(receiveStream, Encoding.UTF8);

		string content = reader.ReadToEnd ();


		AuthResponse o = JsonUtility.FromJson<AuthResponse> (content);
		Debug.Log (o.access_token);

		return o.access_token;


	}




	//some data contract-like class
	class AuthResponse{
		public string access_token;
		public string session_key;
		public string scope;
		public string refresh_token;
		public string session_secret;
		public long expires_in;
	}

	/*
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
	*/


	class RecogRequest{
		public string format;
		public int rate;
		public int channel;
		public string lan;
		public string token;
		public string cuid;
		public long len;
		public string speech;
	}


	class RecogResponse{
		public string corpus_no;
		public string err_msg;
		public int err_no;
		public string sn;
		public string[] result;
	}







}
