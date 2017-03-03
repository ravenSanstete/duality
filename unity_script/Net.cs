using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;




// this module wraps the basic net behaviors

public class Net : MonoBehaviour {

	public static string DEVICE_ID=null;



	//the queues here are thread-safe
	//this queue stores the temp file that is still not uploaded to an online recognition service
	private Queue wav_queue=Queue.Synchronized(new Queue());


	//this queue for the client to store the recognized speech(in a string form)
	private Queue comment_queue=Queue.Synchronized(new Queue());





	//the following queues can be accessed by other modules through some interfaces
	private Queue comment_seq=Queue.Synchronized(new Queue());
	private Queue command_seq=Queue.Synchronized(new Queue());






	// this is for the local http connection
	//private string SERVER_ADDR= "http://192.168.31.44:4000";
	private string SERVER_ADDR= "http://127.0.0.1:4000";
	private int SERVER_PORT=4000;

	private int UDP_PORT=13000;
	private int UDP_BUFFER_SIZE = 4096;


	Thread upload_wav;
	Thread upload_comment;
	Thread listen_server; //this waiting for the server push


	// Use this for initialization
	void Start () {
		//bestow the device id to a public static thing
		DEVICE_ID = SystemInfo.deviceUniqueIdentifier;


		upload_wav = new Thread (new ThreadStart (uploadWav));
		upload_comment = new Thread (new ThreadStart (uploadComment));
		listen_server = new Thread (new ThreadStart (listen));


		Debug.Log ("Start Net Communication");

		upload_wav.Start ();
		upload_comment.Start ();
		listen_server.Start();
		
	}


	// Update is called once per frame, for each update check whether the command seq has something
	void Update () {
		if (command_seq.Count > 0) {
			//invoke some method of MIKU
			/***/
			Protocol.MikuCommand translated_miku_cmd = Command.getInstance ().translate ((Protocol.RawMikuCommand)command_seq.Dequeue ());

			//check both the zeros
			Debug.Log ("Motion ID:" + translated_miku_cmd.motion_id);
			Debug.Log ("Voice ID:" + translated_miku_cmd.voice_id);
			if (translated_miku_cmd.morph_param != null) {
				Debug.Log ("Morph Part:" + translated_miku_cmd.morph_param [0]);
				Debug.Log ("Morph Name:" + translated_miku_cmd.morph_param [1]);
			}
			//* fill in some interface invocation*/

		
			// GameObject.FindGameObjectWithTag("Miku").GetComponent<Animator>().SetInteger("status", translated_miku_cmd.motion_id);
			//

			/* fill in some interface invocation */
		}
	}


	public Protocol.CommentOnFlight[] fetch_comments(){
		int size = comment_seq.Count;
		Protocol.CommentOnFlight [] outcome= new Protocol.CommentOnFlight[size] ;
		int i = 0;
		while (i < size) {
			outcome [i] = (Protocol.CommentOnFlight)comment_seq.Dequeue (); //release those comments that have already been fetched
		}

		return outcome;
	}



	// an UI interface for enqueue the voice data that already been stored temporarily
	public void enqueue_wav_path(string path){
		// Debug.Log ("Enqueue " + path);
		wav_queue.Enqueue (path);
	}





	//this method works on another 
	private void uploadWav(){
		Debug.Log ("Wave thread here.");
		while (true) {
			
			while(wav_queue.Count>0){
				
				string current_file_name = (string)wav_queue.Dequeue ();
				Debug.Log (current_file_name);
				byte[] dat=null;

				try{
					dat = File.ReadAllBytes (current_file_name);

				}catch(IOException ex){
					Debug.Log (ex);
				}

				Debug.Log (dat.Length);
				string content= Sensor.recognize(dat);

				Protocol.CommentOnFlight comment = new Protocol.CommentOnFlight ();
				comment.content = content;
				comment.timestamp = DateTime.Now.ToString(); 
				comment.id = DEVICE_ID;

				comment_queue.Enqueue (comment); //put it into the comment queue

				Debug.Log (content);
				File.Delete(current_file_name); //delete the uploaded data
			}
		}


	}





	private void uploadComment(){
		//here again, to construct a webreq and then pass a json to the server
		while (true) {
			while (comment_queue.Count > 0) {
				Debug.Log ("Post comment to the server");
//				Uri uri = new Uri (SERVER_ADDR);
//				Debug.Log (uri.Port);
				string json_content = JsonUtility.ToJson (comment_queue.Dequeue ());
				Debug.Log (json_content);
				WebRequest req = WebRequest.Create (SERVER_ADDR);
				// Debug.Log (SERVER_ADDR + ":" + SERVER_PORT);


				Debug.Log (json_content);
				req.Method = "POST";
				req.ContentType = "application/json; charset=utf-8";
				req.ContentLength = json_content.Length;
				Debug.Log ("Here");
				Stream req_stream = req.GetRequestStream ();
				Debug.Log ("Here2");
				byte[] byte_content = Encoding.UTF8.GetBytes (json_content);
				// req.ContentLength = byte_content.Length;
				req_stream.Write (byte_content, 0, byte_content.Length);
				req_stream.Flush ();
				req_stream.Close ();


			}
		}
	}



	void onApplicationQuit(){
		if (upload_wav != null) {
			upload_wav.Interrupt ();
			upload_wav.Abort ();
		}

		if (upload_comment != null) {
			upload_comment.Interrupt ();
			upload_comment.Abort ();
		}

		if (listen_server != null) {
			listen_server.Interrupt ();
			listen_server.Abort ();
		}
	}







	//the local udp connection started here
	private void listen(){
		UdpClient server=null;   
		try
		{
			// Set the TcpListener on port 13000.
			server = new UdpClient(UDP_PORT);

			// Start listening for client requests.


			// Buffer for reading data
			byte[] bytes = new byte[UDP_BUFFER_SIZE];
			string data = null;
			int counter = 0;

			IPEndPoint RemoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);
			// Enter the listening loop.
			while(true) 
			{

				// Perform a blocking call to accept requests.
				// You could also user server.AcceptSocket() here.
				bytes= server.Receive(ref RemoteIpEndPoint);
				counter++;
				Debug.Log("#"+counter+" pkt of data been Pushed");

				data = System.Text.Encoding.UTF8.GetString(bytes);

				Debug.Log(data);


				// Process the data sent by the server a.t. the data format we defined
				Protocol.DataPkt pkt= JsonUtility.FromJson<Protocol.DataPkt>(data);
				Debug.Log("here");

				Protocol.RawMikuCommand command= new Protocol.RawMikuCommand();
				command.action_id=pkt.action_id;
				command.voice_id=pkt.voice_id;
				command.option=pkt.option;

				command_seq.Enqueue(command);


				int comment_counter=0;
				Debug.Log("here");
				Debug.Log(pkt.comments);
				foreach(Protocol.CommentOnFlight comment in pkt.comments){
					comment_seq.Enqueue(comment);
					Debug.Log("Get Comment:"+comment.content);
					comment_counter++;
				}

				Debug.Log(comment_counter+" Comments-on-flight Got!");

			}
		}
		catch(SocketException e)
		{
			Debug.Log("SocketException");
		}
		finally
		{
			// Stop listening for new clients.
			server.Close();
		}
	}

















}
