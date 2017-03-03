using System;

namespace Protocol
{
	//this works like a data contract
	[Serializable]
	public class CommentOnFlight{
		public string timestamp;
		public string content;
		public string id; //this can be considered as the unique identity of the system
	}


	[Serializable]
	public class DataPkt{
		public int action_id;
		public int voice_id;
		public int option;
		public CommentOnFlight[] comments;
	}

	[Serializable]
	public class MikuCommand{
		public int voice_id;
		public string[] morph_param;
		public int motion_id;
	}


	[Serializable]
	// which is a primitive class that wraps the original form of data comes from the Server
	public class RawMikuCommand{
		public int voice_id;
		public int action_id;
		public int option;
	
	}

}

