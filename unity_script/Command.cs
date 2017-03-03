using System;
using System.Collections.Generic;





//this works as a data class to maintain four important mappings to parse the command from the server
//you should use it as a singleton
public class Command{
	private Dictionary<int, int> voice_mapping;
	private Dictionary<int, int> motion_mapping;
	private Dictionary<int, string[]> morph_mapping;
	private Dictionary<int, int> song_mapping;



	private static Command c_instance=null;


	public static int SING_DANCE_NO=63;



	public static Command getInstance(){
		if (c_instance == null) {
			c_instance = new Command ();
		}

		return c_instance;
	}


	//critical data
	private Command(){
		voice_mapping = new Dictionary<int, int> ();
		motion_mapping = new Dictionary<int, int> ();
		morph_mapping = new Dictionary<int, string[]> ();
		song_mapping = new Dictionary<int, int> ();

		//to init the mappings
		//begin voice
		voice_mapping.Add(0, 5);
		voice_mapping.Add (1, 4);
		voice_mapping.Add (2, 1);
		voice_mapping.Add (3, 2);
		voice_mapping.Add (4, 3);


		//begin motion
		motion_mapping.Add (1, 1);
		motion_mapping.Add (2, 2);
		motion_mapping.Add (3, 3);
		motion_mapping.Add (4, 4);
		motion_mapping.Add (5, 5);
		motion_mapping.Add (6, 6);
		motion_mapping.Add (7, 7);
		motion_mapping.Add (8, 8);
		motion_mapping.Add (9, 9);
		motion_mapping.Add (10, 10);
		motion_mapping.Add (11, 11);

	
		//begin morph
		morph_mapping.Add(12, tuple("EyeBrow", "goup"));
		morph_mapping.Add(13, tuple("EyeBrow", "godown"));
		morph_mapping.Add(14, tuple("EyeBrow", "anger"));
		morph_mapping.Add(15, tuple("Eye", "blink"));
		morph_mapping.Add (16, tuple ("Eye", "smile"));
		morph_mapping.Add(17, tuple("Eye", "wink"));
		morph_mapping.Add(18, tuple("Eye", "wink_R"));
		morph_mapping.Add(19, tuple("Eye", "close><"));
		morph_mapping.Add(20, tuple("Eye", "whiteeye"));
		morph_mapping.Add(21, tuple("Lip", "a"));
		morph_mapping.Add(22, tuple("Lip", "i"));
		morph_mapping.Add(23, tuple("Lip", "u"));
		morph_mapping.Add(24, tuple("Lip", "o"));
		morph_mapping.Add(25, tuple("Lip", "regret"));
		morph_mapping.Add(26, tuple("Lip", "box"));
		morph_mapping.Add(27, tuple("Other", "cheek"));
		morph_mapping.Add(28, tuple("Other", "cheekB"));
		morph_mapping.Add(29, tuple("Other", "tear"));
		morph_mapping.Add(30, tuple("Clear", "clear"));







		//begin song name
		song_mapping.Add(1,12);
		song_mapping.Add(2, 13);
		song_mapping.Add (3, 14);
		song_mapping.Add (4, 15);
		song_mapping.Add (5, 16);
		song_mapping.Add (6, 17);
	}






	public Protocol.MikuCommand translate(Protocol.RawMikuCommand raw){
		Protocol.MikuCommand cmd = new Protocol.MikuCommand ();

		int action_id = 0;
		string[] morph_params = null;

		//if it is singing or dancing 
		if (raw.action_id == SING_DANCE_NO) {
			//prevent some bad search
			if (song_mapping.ContainsKey (raw.option)) {
				action_id = song_mapping [raw.option];
			}
		
		//if it is some motion-action
		} else if (motion_mapping.ContainsKey (raw.action_id)) {
			action_id = motion_mapping [raw.action_id];
		//if it is some morph action
		} else if (morph_mapping.ContainsKey (raw.action_id)) {
			morph_params = morph_mapping [raw.action_id];
		} // an omit else means nothing happens



		cmd.voice_id = raw.voice_id;
		cmd.morph_param = morph_params;
		cmd.motion_id = action_id;

		 
		return cmd;
	}





	private string[] tuple(string morph_part, string morph_name){
		string[] x = new string[2];
		x [0] = morph_part;
		x [1] = morph_name;
		return x;
	}









}

