using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Net;
using System.IO;
using UnityEngine;

[RequireComponent (typeof(AudioSource))]
public class SpeechIn : MonoBehaviour {

	//construct it as a singleton
	private static SpeechIn speech_instance=null;


	private static string [] device_arr=null;



	private static int RECORD_TIME = 3;


	private static int RATE = 8000;

	private static string PREFIX=null;








	// Use this for initialization
	void Start () {


		PREFIX = "/Users/morino/Desktop/tmp/"; //the temporary data storage
	}


	public static SpeechIn getInstance(){
		if (speech_instance == null) {
			device_arr = Microphone.devices;
			if (device_arr.Length == 0) {
				Debug.Log("No mircophone device");
				return null;
			}

			foreach (string deviceStr in device_arr)  
			{  
				Debug.Log("device name: " + deviceStr);  
			}  

			GameObject speech_in = new GameObject ("speech_in");
			speech_instance= speech_in.AddComponent<SpeechIn> ();


		}

		return speech_instance;
		
	}


	public void startRecord()  
	{  
		AudioSource src = this.GetComponent<AudioSource> ();
		src.Stop();  
		if (device_arr.Length == 0)  
		{  
			Debug.Log("No Record Device!");  
			return;  
		}  
		src.loop = false;  
		src.mute = true;  
		src.clip = Microphone.Start(null, false, RECORD_TIME, RATE);   
		while (!(Microphone.GetPosition(null)>0)) {  
		}  
		src.Play ();  

		Debug.Log("StartRecord");  
		 

	}  





	//return the audio clip stored path or null(if no record currently)
	public string stopRecord()  
	{  
		if (device_arr.Length == 0)  
		{  
			Debug.Log("No Record Device!");  
			return null; 
		}  
		if (!Microphone.IsRecording(null))  
		{  
			return null;
		}  



		AudioSource src = this.GetComponent<AudioSource> ();
		Microphone.End (null);  
		src.Stop();  


	

		Debug.Log("StopRecord");

		// Thread save_wav_thread = new Thread (new ParameterizedThreadStart (save_tmp_wav));

		if (src.clip != null) {
			float[] samples = new float[src.clip.samples];
			src.clip.GetData (samples, 0);

			//construct the data for the wave-saver thread usage
			ThreadData data = new ThreadData ();

			data.samples = samples;
			data.samples_num=src.clip.samples;
			data.hz= src.clip.frequency;
			data.channels = src.clip.channels;
			data.save_path= PREFIX+System.Guid.NewGuid()+".wav";

			save_tmp_wav (data);

			return data.save_path;
		}else {
			Debug.Log ("Nothing Recorded");
			return null;
		}




	}  



	class ThreadData{
		public string save_path;
		public float[] samples;
		public int hz;
		public int channels;
		public int samples_num;
	}




	//for a new thread to invoke it, thus not intercept the ui thread
	private void save_tmp_wav(object maybe){

		ThreadData obj = (ThreadData)maybe;
		SavWav.Save (obj.save_path, obj.samples, obj.hz, obj.channels, obj.samples_num);
	}







	// Update is called once per frame
	void Update () {
		
	}





}
