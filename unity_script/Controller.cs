using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;


public class Controller : MonoBehaviour {

	private Button start_button;
	private Button stop_button;
	private Button play_button;
	private Voice voice;


	// Use this for initialization
	void Start () {
		GameObject start_obj = GameObject.Find ("Begin");
		GameObject stop_obj = GameObject.Find ("Stop");
		GameObject play_obj = GameObject.Find ("Play");


		voice = GameObject.Find ("AudioPlayer").GetComponent<Voice> ();
		start_button = start_obj.GetComponent<Button> ();
		stop_button = stop_obj.GetComponent<Button> ();
		play_button = play_obj.GetComponent<Button> ();



		start_button.onClick.AddListener (startEvent);
		stop_button.onClick.AddListener (stopEvent);
		play_button.onClick.AddListener (playEvent);
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}


	void startEvent(){
		Debug.Log ("Begin be clicked");
		SpeechIn.getInstance ().startRecord ();
		Debug.Log ("Begin be clicked");
	}


	void stopEvent(){
		
		GameObject.Find("Canvas").GetComponent<Net>().enqueue_wav_path(SpeechIn.getInstance ().stopRecord ());

		Debug.Log ("Stop be clicked");
	}

	void playEvent(){
		voice.Play ("Assets/audio/permission.mp3");
		Debug.Log ("Play be clicked");
	}








}
