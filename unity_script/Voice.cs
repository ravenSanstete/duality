using System.Collections;
using System.Collections.Generic;
using UnityEngine;





// this wraps the voice switcher module
public class Voice : MonoBehaviour {

	private AudioSource src;



	public void Play(string voice_no){
		src.clip = (AudioClip)Resources.Load (voice_no, typeof(AudioClip));
		src.Play ();
	}

	public void Stop(){
		src.Stop ();
	}



	// Use this for initialization
	void Start () {
		src = GetComponent<AudioSource> ();
	}



	
	// Update is called once per frame
	void Update () {
		
	}


}
