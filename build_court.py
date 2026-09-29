"""Reproducible VoiceSkip Fold7 GigaAM Android source and verified model preparation.
No private recordings or transcripts are used by this build.
"""
from pathlib import Path
import hashlib, json, urllib.request, tarfile, concurrent.futures

SOURCES = {
'app/build.gradle.kts': r'''plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "com.voiceskip.fold7.court"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.voiceskip.fold7.court"
        minSdk = 35
        targetSdk = 35
        versionCode = 100
        versionName = "1.0-GigaAM"
        ndk { abiFilters += if (project.hasProperty("emulatorTest")) listOf("x86_64") else listOf("arm64-v8a") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildTypes { getByName("release") { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("debug") } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += listOf("onnx", "bin") }
    packaging { jniLibs { useLegacyPackaging = false } }
}
dependencies {
    implementation(files("libs/sherpa.aar"))
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:core:1.6.1")
}
''',

'app/src/androidTest/java/com/voiceskip/fold7/court/DeviceTest.kt': r'''package com.voiceskip.fold7.court

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.core.app.ActivityScenario
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import com.k2fsa.sherpa.onnx.*
import android.net.Uri
import java.io.File
import java.io.ByteArrayOutputStream

@RunWith(AndroidJUnit4::class)
class DeviceTest {
 @Test fun nativeModelsDecodeAudioAndAttributeRealSpeech() {
    val instrumentation=InstrumentationRegistry.getInstrumentation()
    val context=instrumentation.targetContext
    val wav=File(context.cacheDir,"public-example.wav")
    instrumentation.context.assets.open("example.wav").use{input->wav.outputStream().use{input.copyTo(it)}}
    val pcm=File(context.cacheDir,"test.pcm")
    Audio.decode(context,Uri.fromFile(wav),pcm,{}, {false})
    val samples=Audio.read(pcm)
    assertTrue(samples.size>16000)
    val r=OfflineRecognizer(context.assets,OfflineRecognizerConfig(modelConfig=OfflineModelConfig(
       transducer=OfflineTransducerModelConfig(encoder="models/gigaam_v3_e2e_rnnt_encoder_int8.onnx",decoder="models/gigaam_v3_e2e_rnnt_decoder.onnx",joiner="models/gigaam_v3_e2e_rnnt_joint.onnx"),tokens="models/gigaam_v3_e2e_rnnt_tokens.txt",modelType="nemo_transducer",numThreads=2)))
    val stream=r.createStream()
    val words:List<Word>
    try {
      stream.acceptWaveform(samples.copyOf(samples.size+16000),16000);r.decode(stream)
      val result=r.getResult(stream)
      assertTrue("Russian speech must be decoded",result.text.count{it in 'а'..'я' || it in 'А'..'Я'}>15)
      assertEquals(result.tokens.size,result.timestamps.size)
      words=Transcript.words(result.tokens,result.timestamps,0f,samples.size/16000f)
      assertTrue(words.size>5)
    }finally{stream.release();r.release()}
    val d=OfflineSpeakerDiarization(context.assets,OfflineSpeakerDiarizationConfig(
      segmentation=OfflineSpeakerSegmentationModelConfig(pyannote=OfflineSpeakerSegmentationPyannoteModelConfig(model="models/segmentation.onnx"),numThreads=2),
      embedding=SpeakerEmbeddingExtractorConfig(model="models/embedding.onnx",numThreads=2),clustering=FastClusteringConfig(threshold=.9f)))
    val turns=try{d.process(samples).map{Turn(it.start,it.end,it.speaker)}}finally{d.release()}
    assertTrue("Speech must have diarization intervals",turns.isNotEmpty())
    val blocks=Transcript.blocks(words,turns)
    assertTrue(blocks.any{it.speaker>=0})
    val session=Session("000-test","Проверочная запись",true,blocks,raw=words.joinToString(" "){it.text});session.save(context)
    val restored=Session.load(File(context.filesDir,"sessions/000-test.json"));assertEquals(blocks.size,restored.blocks.size)
    Transcript.docx(ByteArrayOutputStream(),restored.blocks,restored::name)
    ActivityScenario.launch(MainActivity::class.java).use{scenario->scenario.onActivity{assertNotNull(it.window.decorView)}}
 }
}
''',

'app/src/main/AndroidManifest.xml': r'''<manifest xmlns:android="http://schemas.android.com/apk/res/android">
  <uses-permission android:name="android.permission.RECORD_AUDIO"/>
  <uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
  <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>
  <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROCESSING"/>
  <uses-permission android:name="android.permission.WAKE_LOCK"/>
  <uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
  <application android:theme="@android:style/Theme.Material.Light.NoActionBar" android:label="VoiceSkip Fold7 GigaAM" android:allowBackup="false" android:largeHeap="true" android:supportsRtl="true">
    <activity android:name=".MainActivity" android:exported="true" android:configChanges="orientation|screenSize|smallestScreenSize">
      <intent-filter><action android:name="android.intent.action.MAIN"/><category android:name="android.intent.category.LAUNCHER"/></intent-filter>
      <intent-filter><action android:name="android.intent.action.SEND"/><category android:name="android.intent.category.DEFAULT"/><data android:mimeType="audio/*"/></intent-filter>
    </activity>
    <service android:name=".TranscriptionService" android:exported="false" android:foregroundServiceType="microphone|mediaProcessing"/>
  </application>
</manifest>
''',

'app/src/main/assets/NOTICE.txt': r'''GigaAM v3: MIT, SberDevices / salute-developers.
https://github.com/salute-developers/GigaAM
ONNX conversion: sherpa-onnx; pinned mirror pantinor/gigaam-v3.
sherpa-onnx: Apache-2.0, Xiaomi Corporation / k2-fsa.
https://github.com/k2-fsa/sherpa-onnx
ONNX Runtime: MIT, Microsoft Corporation.
https://github.com/microsoft/onnxruntime
Pyannote segmentation 3.0: MIT, pyannote.
https://huggingface.co/pyannote/segmentation-3.0
3D-Speaker ERes2Net: Apache-2.0, Alibaba.
https://github.com/modelscope/3D-Speaker
Полные лицензии включены в assets/licenses.
''',

'app/src/main/java/com/voiceskip/fold7/court/Audio.kt': r'''package com.voiceskip.fold7.court

import android.content.Context
import android.media.*
import android.net.Uri
import java.io.*
import java.nio.ByteOrder

object Audio {
    fun decode(context: Context, uri: Uri, output: File, progress: (String) -> Unit, cancelled: () -> Boolean) {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
                ?: error("В файле нет аудиодорожки")
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val duration = if(format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0
            require(duration <= 2L * 60 * 60 * 1000000) { "Запись длиннее двух часов. Разделите её на части." }
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec.configure(format, null, null, 0); codec.start()
            val info = MediaCodec.BufferInfo()
            var inputEnd = false; var outputEnd = false
            var inputIndex = 0L; var targetIndex = 0L; var previous = 0f; var count = 0L
            DataOutputStream(BufferedOutputStream(FileOutputStream(output))).use { sink ->
                while (!outputEnd) {
                    check(!cancelled()) { "Обработка отменена" }
                    if (!inputEnd) {
                        val index = codec.dequeueInputBuffer(10000)
                        if (index >= 0) {
                            val buffer = codec.getInputBuffer(index)!!
                            val size = extractor.readSampleData(buffer, 0)
                            if (size < 0) { codec.queueInputBuffer(index,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputEnd=true }
                            else { codec.queueInputBuffer(index,0,size,extractor.sampleTime,0); extractor.advance() }
                        }
                    }
                    val index = codec.dequeueOutputBuffer(info,10000)
                    if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val f=codec.outputFormat
                        val newRate=f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        require(inputIndex==0L || rate==newRate) { "Частота звука меняется внутри файла" }
                        rate=newRate;channels=f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        encoding=if(f.containsKey(MediaFormat.KEY_PCM_ENCODING))f.getInteger(MediaFormat.KEY_PCM_ENCODING) else AudioFormat.ENCODING_PCM_16BIT
                        require(encoding==AudioFormat.ENCODING_PCM_16BIT || encoding==AudioFormat.ENCODING_PCM_FLOAT) { "Неподдерживаемая разрядность аудио" }
                    } else if (index>=0) {
                        val b=codec.getOutputBuffer(index)!!.order(ByteOrder.LITTLE_ENDIAN)
                        b.position(info.offset);b.limit(info.offset+info.size)
                        val bytes=if(encoding==AudioFormat.ENCODING_PCM_FLOAT)4 else 2
                        while(b.remaining()>=bytes*channels) {
                            var sample=0f
                            repeat(channels) { sample+=if(bytes==4)b.float else b.short/32768f };sample/=channels
                            // Stateful interpolation: no sample loss at codec buffer boundaries.
                            while(targetIndex*rate.toDouble()/16000 <= inputIndex) {
                                val x=targetIndex*rate.toDouble()/16000
                                val fraction=(x-(inputIndex-1)).coerceIn(0.0,1.0).toFloat()
                                sink.writeFloat(previous+(sample-previous)*fraction);targetIndex++;count++
                                require(count<=2L*60*60*16000) { "Запись длиннее двух часов" }
                            }
                            previous=sample;inputIndex++
                        }
                        outputEnd=info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM !=0
                        codec.releaseOutputBuffer(index,false)
                        if(duration>0)progress("Подготовка аудио: ${(info.presentationTimeUs*100/duration).coerceIn(0,100)}%")
                    }
                }
            }
            require(count>0) { "Аудиодорожка пуста" }
        } finally { runCatching { codec?.stop() };codec?.release();extractor.release() }
    }
    fun read(file: File): FloatArray {
        val count=(file.length()/4).toInt()
        val runtime=Runtime.getRuntime()
        require(file.length() < (runtime.maxMemory()-runtime.totalMemory()+runtime.freeMemory())*.7) { "Недостаточно памяти для этой записи. Обработайте её частями." }
        val samples=FloatArray(count)
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input -> for(i in samples.indices)samples[i]=input.readFloat() }
        return samples
    }
}
''',

'app/src/main/java/com/voiceskip/fold7/court/MainActivity.kt': r'''package com.voiceskip.fold7.court

import android.Manifest
import android.app.*
import android.content.*
import android.content.pm.PackageManager
import android.media.*
import android.net.Uri
import android.os.*
import android.provider.OpenableColumns
import android.graphics.Color
import android.view.*
import android.widget.*
import java.io.*

class MainActivity: Activity() {
    private lateinit var root:LinearLayout
    private lateinit var status:TextView
    private lateinit var rows:LinearLayout
    private lateinit var record:Button
    private lateinit var open:Button
    private val handler=Handler(Looper.getMainLooper())
    private var session:Session?=null
    private var exportKind="docx"
    private var lastBusy=false
    private var playing=false
    private var player:AudioTrack?=null
    private val refresh=object:Runnable {
        override fun run() {
            status.text=JobState.status
            open.isEnabled=!JobState.busy
            record.isEnabled=!JobState.busy || JobState.recording
            record.text=if(JobState.recording)"Остановить и расшифровать" else "Записать заседание"
            if(lastBusy && !JobState.busy) {
                val file=File(filesDir,"sessions/${JobState.sessionId}.json")
                if(file.exists()){session=Session.load(file);render()}
                else if(JobState.status.startsWith("Ошибка")) showError(JobState.status)
            }
            lastBusy=JobState.busy;handler.postDelayed(this,700)
        }
    }
    override fun onCreate(saved:Bundle?) {
        super.onCreate(saved)
        window.statusBarColor=Color.rgb(245,247,251);window.navigationBarColor=Color.rgb(245,247,251)
        root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setBackgroundColor(Color.rgb(245,247,251));setPadding(20,24,20,16)}
        setContentView(root)
        root.setOnApplyWindowInsetsListener{v,insets -> val bars=insets.getInsets(WindowInsets.Type.systemBars());v.setPadding(20,bars.top+12,20,bars.bottom+8);insets}
        title("VoiceSkip Fold7",25)
        title("GigaAM · русский язык · работает без интернета",13)
        status=title(JobState.status,15)
        open=button(root,"Открыть аудиозапись") { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).setType("audio/*").addCategory(Intent.CATEGORY_OPENABLE).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION),10) }
        record=button(root,"Записать заседание") {
            if(JobState.recording)startService(Intent(this,TranscriptionService::class.java).setAction("stop-record"))
            else if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED)requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO),11)
            else startRecording()
        }
        val toolbar=LinearLayout(this);root.addView(toolbar)
        button(toolbar,"Записи") { history() };button(toolbar,"Экспорт") { exportMenu() };button(toolbar,"Ещё") { more() }
        val scroll=ScrollView(this);root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        rows=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL};scroll.addView(rows)
        session=Session.files(this).firstOrNull()?.let{runCatching{Session.load(it)}.getOrNull()}
        render()
        if(Build.VERSION.SDK_INT>=33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS),12)
        if(intent.action==Intent.ACTION_SEND) {
            @Suppress("DEPRECATION") val uri=intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
            if(uri!=null && !JobState.busy)startFile(uri)
        }
    }
    override fun onResume(){super.onResume();handler.post(refresh)}
    override fun onPause(){handler.removeCallbacks(refresh);super.onPause()}
    override fun onDestroy(){stopPlayback();super.onDestroy()}
    private fun title(text:String,size:Int):TextView=TextView(this).apply{this.text=text;textSize=size.toFloat();setTextColor(Color.rgb(28,39,57));setPadding(4,6,4,6);root.addView(this)}
    private fun button(parent:LinearLayout,text:String,action:()->Unit)=Button(this).apply{this.text=text;isAllCaps=false;setOnClickListener{action()};parent.addView(this,if(parent.orientation==LinearLayout.HORIZONTAL)LinearLayout.LayoutParams(0,-2,1f) else LinearLayout.LayoutParams(-1,-2))}
    private fun startRecording(){stopPlayback();startForegroundService(Intent(this,TranscriptionService::class.java).setAction("record"));lastBusy=true}
    private fun startFile(uri:Uri) {
        stopPlayback()
        val name=runCatching{contentResolver.query(uri,arrayOf(OpenableColumns.DISPLAY_NAME),null,null,null)?.use{if(it.moveToFirst())it.getString(0) else null}}.getOrNull()?:"Аудиозапись"
        startForegroundService(Intent(this,TranscriptionService::class.java).setAction("file").setData(uri).putExtra("title",name).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION));lastBusy=true
    }
    override fun onRequestPermissionsResult(requestCode:Int,permissions:Array<out String>,grantResults:IntArray){super.onRequestPermissionsResult(requestCode,permissions,grantResults);if(requestCode==11 && grantResults.firstOrNull()==PackageManager.PERMISSION_GRANTED)startRecording()}
    @Deprecated("Activity result API") override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
        super.onActivityResult(requestCode,resultCode,data)
        if(resultCode!=RESULT_OK)return
        val uri=data?.data?:return
        if(requestCode==10){runCatching{contentResolver.takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION)};startFile(uri)}
        if(requestCode==20)try {
            contentResolver.openOutputStream(uri)?.use{out ->
                when(exportKind){
                    "docx" -> {val s=session?:error("Нет текста");Transcript.docx(out,s.blocks,s::name)}
                    "raw" -> out.write((session?.raw?:"").toByteArray())
                    "txt" -> {val s=session?:error("Нет текста");out.write(s.blocks.joinToString("\n\n"){s.name(it.speaker)+"\n"+it.text}.toByteArray())}
                    "log" -> out.write(File(filesDir,"diagnostics.txt").takeIf{it.exists()}?.readBytes()?:"Ошибок не зарегистрировано".toByteArray())
                }
            }?:error("Не удалось открыть файл")
            Toast.makeText(this,"Файл сохранён",Toast.LENGTH_LONG).show()
        }catch(e:Exception){showError("Не удалось сохранить: ${e.message}")}
    }
    private fun render() {
        rows.removeAllViews();val s=session
        if(s==null){rows.addView(TextView(this).apply{text="Откройте запись или включите микрофон. Говорящие будут разделены автоматически. После обработки можно указать их имена и сохранить документ Word.";textSize=17f;setPadding(8,24,8,8)});return}
        rows.addView(TextView(this).apply{text=s.title+if(s.complete)"" else "\nНезавершённая расшифровка — сохранённая часть";textSize=18f;setPadding(8,18,8,18)})
        s.blocks.forEachIndexed { i,b ->
            val container=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(12,8,12,18);setBackgroundColor(Color.WHITE)}
            rows.addView(container,LinearLayout.LayoutParams(-1,-2).apply{bottomMargin=12})
            val heading=TextView(this).apply{text=s.name(b.speaker);textSize=16f;setTextColor(Color.rgb(30,80,150));setTypeface(null,1);setPadding(0,8,0,8);setOnClickListener{rename(b.speaker)}}
            container.addView(heading)
            container.addView(TextView(this).apply{text=b.text;textSize=17f;setTextColor(Color.rgb(25,30,40));setLineSpacing(3f,1.1f);setOnClickListener{edit(i)}})
            val actions=LinearLayout(this);container.addView(actions)
            button(actions,"▶ Слушать") { play(b.start) }
            button(actions,"Править") { edit(i) }
        }
    }
    private fun edit(index:Int) {
        val s=session?:return;val b=s.blocks[index]
        val panel=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(24,8,24,8)}
        val speaker=Spinner(this);val ids=(s.blocks.map{it.speaker}+listOf(-1)).distinct().sorted()
        speaker.adapter=ArrayAdapter(this,android.R.layout.simple_spinner_dropdown_item,ids.map{s.name(it)})
        speaker.setSelection(ids.indexOf(b.speaker));panel.addView(speaker)
        val field=EditText(this).apply{setText(b.text);minLines=4;maxLines=12;inputType=android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE};panel.addView(field)
        AlertDialog.Builder(this).setTitle("Исправить реплику").setView(panel).setNegativeButton("Отмена",null).setPositiveButton("Сохранить"){_,_->b.text=field.text.toString();b.speaker=ids[speaker.selectedItemPosition];s.save(this);render()}.show()
    }
    private fun rename(id:Int) {
        val s=session?:return;val field=EditText(this).apply{setText(s.name(id));selectAll()}
        AlertDialog.Builder(this).setTitle("Имя или роль говорящего").setMessage("Например: судья, истец, представитель. Применяется ко всем репликам этого говорящего.").setView(field).setNegativeButton("Отмена",null).setPositiveButton("Сохранить"){_,_->val name=field.text.toString().trim();if(name.isNotEmpty()){s.names[id]=name;s.save(this);render()}}.show()
    }
    private fun history() {
        val files=Session.files(this);val sessions=files.mapNotNull{runCatching{Session.load(it)}.getOrNull()}
        AlertDialog.Builder(this).setTitle("Сохранённые записи").setItems(sessions.map{it.title}.toTypedArray()){_,i->session=sessions[i];stopPlayback();render()}.setNegativeButton("Закрыть",null).show()
    }
    private fun exportMenu(){if(session==null){showError("Сначала расшифруйте запись");return};AlertDialog.Builder(this).setTitle("Сохранить файл").setItems(arrayOf("Документ Word (.docx)","Текст со спикерами (.txt)","Исходное распознавание (.txt)")){_,i->export(arrayOf("docx","txt","raw")[i])}.show()}
    private fun export(kind:String) {
        exportKind=kind
        val filename=if(kind=="log")"VoiceSkip-diagnostics.txt" else "Расшифровка-${session?.id}.${if(kind=="docx")"docx" else "txt"}"
        startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType(if(kind=="docx")"application/vnd.openxmlformats-officedocument.wordprocessingml.document" else "text/plain").putExtra(Intent.EXTRA_TITLE,filename),20)
    }
    private fun more(){AlertDialog.Builder(this).setItems(arrayOf("Остановить обработку","Остановить прослушивание","Экспорт ошибки","О приложении")){_,i->when(i){0->JobState.cancel=true;1->stopPlayback();2->export("log");3->AlertDialog.Builder(this).setTitle("VoiceSkip Fold7 GigaAM").setMessage("GigaAM v3 E2E RNNT • sherpa-onnx 1.13.8\nPyannote 3.0 + 3D-Speaker\n\nПунктуация расставляется моделью распознавания. Юридические факты не дописываются. Фамилии, числа и распределение реплик следует сверить с записью. Нажмите на имя для переименования, на текст — для правки.\n\nВсе модели находятся в APK. Интернет не используется.\n\n"+assets.open("NOTICE.txt").bufferedReader().use{it.readText()}).setPositiveButton("Закрыть",null).show()}}.show()}
    private fun showError(message:String){AlertDialog.Builder(this).setTitle("VoiceSkip Fold7").setMessage(message).setPositiveButton("Закрыть",null).show()}
    private fun stopPlayback(){playing=false;runCatching{player?.pause()};player=null}
    private fun play(start:Float) {
        stopPlayback();val s=session?:return;val file=File(filesDir,"audio/${s.id}.pcm")
        if(!file.exists()){showError("Аудиозапись недоступна");return}
        playing=true
        Thread {
            val track=AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()).setAudioFormat(AudioFormat.Builder().setSampleRate(16000).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).build()).setBufferSizeInBytes(64000).build()
            player=track
            try {
                RandomAccessFile(file,"r").use{input -> input.seek((maxOf(0f,start-.3f)*16000).toLong()*4);track.play();val b=FloatArray(4000)
                    while(playing && player===track && input.filePointer<input.length()) {val n=minOf(b.size.toLong(),(input.length()-input.filePointer)/4).toInt();for(i in 0 until n)b[i]=input.readFloat();track.write(b,0,n,AudioTrack.WRITE_BLOCKING)}
                }
            }catch(_:Exception){}finally{track.release();if(player===track)player=null}
        }.start()
    }
}
''',

'app/src/main/java/com/voiceskip/fold7/court/Session.kt': r'''package com.voiceskip.fold7.court
import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class Session(val id: String, var title: String, var complete: Boolean=false,
              val blocks: MutableList<Block> = mutableListOf(), val names: MutableMap<Int,String> = mutableMapOf(), var raw: String="") {
    fun name(id: Int) = names[id] ?: if(id<0) "Говорящий не определён" else "Говорящий ${id+1}"
    fun save(context: Context) {
        val j=JSONObject().put("id",id).put("title",title).put("complete",complete).put("raw",raw)
        j.put("blocks",JSONArray().also { a -> blocks.forEach { a.put(JSONObject().put("start",it.start).put("speaker",it.speaker).put("text",it.text)) } })
        j.put("names",JSONObject().also { o -> names.forEach { (k,v)->o.put(k.toString(),v) } })
        val dir=File(context.filesDir,"sessions").also { it.mkdirs() };val tmp=File(dir,"$id.tmp")
        tmp.writeText(j.toString());check(tmp.renameTo(File(dir,"$id.json"))) { "Не удалось сохранить текст" }
    }
    companion object {
        fun load(file: File): Session {
            val j=JSONObject(file.readText());val s=Session(j.getString("id"),j.getString("title"),j.getBoolean("complete"),raw=j.getString("raw"))
            val a=j.getJSONArray("blocks");for(i in 0 until a.length()){val b=a.getJSONObject(i);s.blocks.add(Block(b.getDouble("start").toFloat(),b.getInt("speaker"),b.getString("text")))}
            val n=j.getJSONObject("names");n.keys().forEach { s.names[it.toInt()]=n.getString(it) };return s
        }
        fun files(context: Context)=File(context.filesDir,"sessions").listFiles()?.filter { it.extension=="json" }?.sortedByDescending { it.name } ?: emptyList()
    }
}

object JobState {
    @Volatile var busy=false
    @Volatile var recording=false
    @Volatile var cancel=false
    @Volatile var stopRecord=false
    @Volatile var status="Готово к работе"
    @Volatile var sessionId:String?=null
}
''',

'app/src/main/java/com/voiceskip/fold7/court/Transcript.kt': r'''package com.voiceskip.fold7.court

import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

data class Turn(val start: Float, val end: Float, val speaker: Int)
data class Word(val start: Float, val end: Float, val text: String)
data class Block(val start: Float, var speaker: Int, var text: String)

object Transcript {
    fun words(tokens: Array<String>, times: FloatArray, offset: Float, end: Float): List<Word> {
        require(tokens.size == times.size) { "Модель вернула неполные временные метки" }
        val out = mutableListOf<Word>()
        var text = ""; var start = offset; var last = offset
        for (i in tokens.indices) {
            val token = tokens[i].replace("▁", " ")
            val t = (offset + times[i]).coerceIn(offset, end)
            if (token.startsWith(" ") && text.isNotBlank()) {
                out.add(Word(start, maxOf(start + .02f, minOf(t, last + .25f)).coerceAtMost(end), text.trim()))
                text = ""
            }
            if (text.isEmpty()) start = t
            text += token; last = t
        }
        if (text.isNotBlank()) out.add(Word(start, minOf(end, last + .2f), text.trim()))
        return out
    }
    fun speaker(word: Word, turns: List<Turn>): Int {
        val duration = maxOf(.02f, word.end - word.start)
        val scores = turns.groupBy { it.speaker }.mapValues { (_, rows) ->
            // Union intervals prevents overlapping windows from inflating confidence.
            var total = 0f; var right = word.start
            for (t in rows.sortedBy { it.start }) {
                val left = maxOf(word.start, t.start, right); val end = minOf(word.end, t.end)
                if (end > left) { total += end - left; right = end }
            }; total
        }.entries.sortedByDescending { it.value }
        if (scores.isEmpty() || scores[0].value < duration * .65f) return -1
        if (scores.size > 1 && scores[1].value > duration * .2f) return -1
        return scores[0].key
    }
    fun blocks(words: List<Word>, turns: List<Turn>): MutableList<Block> {
        val out = mutableListOf<Block>()
        var previousEnd = 0f
        for (word in words) {
            val id = speaker(word, turns)
            val last = out.lastOrNull()
            if (last != null && last.speaker == id && word.start - previousEnd < 2f && last.text.length < 1000) {
                last.text += if (word.text.firstOrNull() in listOf('.', ',', ':', ';', '?', '!')) word.text else " ${word.text}"
            } else out.add(Block(word.start, id, word.text))
            previousEnd = word.end
        }
        return out
    }
    fun xml(s: String) = s.filter { it == '\n' || it == '\t' || it >= ' ' }
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
    fun docx(output: OutputStream, blocks: List<Block>, name: (Int) -> String) {
        ZipOutputStream(output).use { zip ->
            fun entry(path: String, text: String) { zip.putNextEntry(ZipEntry(path)); zip.write(text.toByteArray()); zip.closeEntry() }
            entry("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""")
            entry("_rels/.rels", """<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>""")
            fun p(text: String, bold: Boolean = false) = "<w:p><w:pPr><w:spacing w:after=\"140\" w:line=\"300\"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\" w:cs=\"Times New Roman\"/><w:sz w:val=\"24\"/><w:lang w:val=\"ru-RU\"/>${if (bold) "<w:b/>" else ""}</w:rPr><w:t xml:space=\"preserve\">${xml(text)}</w:t></w:r></w:p>"
            val body = buildString {
                append(p("Расшифровка судебного заседания", true))
                append(p("Автоматическая расшифровка аудиозаписи. Не является официальным протоколом."))
                blocks.forEach { append(p(name(it.speaker), true)); append(p(it.text)) }
            }
            entry("word/document.xml", """<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>$body<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1701"/></w:sectPr></w:body></w:document>""")
        }
    }
    fun chunkEnd(samples: FloatArray, start: Int, rate: Int = 16000): Int {
        val cap = minOf(start + 25 * rate, samples.size)
        if (cap == samples.size) return cap
        // Leave >= 3 s in the last window. Every sample belongs to exactly one window.
        val earliest = start + 20 * rate
        val latest = minOf(cap, samples.size - 3 * rate)
        var best = latest; var energy = Double.MAX_VALUE
        val half = rate / 20
        for (center in earliest..latest step rate / 10) {
            var e = 0.0
            for (i in center - half until center + half) e += samples[i] * samples[i]
            if (e < energy) { energy = e; best = center }
        }
        return best
    }
}
''',

'app/src/main/java/com/voiceskip/fold7/court/TranscriptionService.kt': r'''package com.voiceskip.fold7.court

import android.app.*
import android.content.*
import android.content.pm.ServiceInfo
import android.media.*
import android.os.*
import com.k2fsa.sherpa.onnx.*
import java.io.*
import java.security.MessageDigest
import org.json.JSONArray

class TranscriptionService: Service() {
    private var wake:PowerManager.WakeLock?=null
    override fun onBind(intent:Intent?)=null
    override fun onStartCommand(intent:Intent?, flags:Int, startId:Int):Int {
        if(intent==null)return START_NOT_STICKY
        if(intent.action=="stop-record"){JobState.stopRecord=true;return START_NOT_STICKY}
        if(JobState.busy)return START_NOT_STICKY
        JobState.busy=true;JobState.cancel=false;JobState.stopRecord=false;JobState.recording=intent.action=="record"
        val manager=getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("work","Запись и расшифровка",NotificationManager.IMPORTANCE_LOW))
        val pending=PendingIntent.getActivity(this,0,Intent(this,MainActivity::class.java),PendingIntent.FLAG_IMMUTABLE)
        val notification=Notification.Builder(this,"work").setSmallIcon(android.R.drawable.ic_btn_speak_now).setContentTitle("VoiceSkip Fold7").setContentText(if(JobState.recording)"Идёт запись" else "Обработка аудио").setContentIntent(pending).setOngoing(true).build()
        val type=if(JobState.recording) ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else if(Build.VERSION.SDK_INT>=35)ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING else 0
        startForeground(1,notification,type)
        wake=(getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"VoiceSkip:transcription").also { it.acquire(6*60*60*1000L) }
        Thread {
            try {
                val id=System.currentTimeMillis().toString(); JobState.sessionId=id
                val session=Session(id,intent.getStringExtra("title")?:"Запись ${java.text.SimpleDateFormat("dd.MM.yyyy HH:mm",java.util.Locale.ROOT).format(java.util.Date())}")
                val audioDir=File(filesDir,"audio").also { it.mkdirs() };val pcm=File(audioDir,"$id.pcm")
                if(JobState.recording) {
                    record(pcm)
                    JobState.recording=false
                    startForeground(1,Notification.Builder(this,"work").setSmallIcon(android.R.drawable.ic_btn_speak_now).setContentTitle("VoiceSkip Fold7").setContentText("Обработка записи").setContentIntent(pending).build(),if(Build.VERSION.SDK_INT>=35)ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING else 0)
                } else Audio.decode(this,intent.data?:error("Не выбран файл"),pcm,{JobState.status=it},{JobState.cancel})
                check(!JobState.cancel){"Обработка отменена"}
                JobState.status="Подготовка моделей"
                val modelDir=prepareModels()
                val samples=Audio.read(pcm)
                process(samples,modelDir,session)
                JobState.status="Готово · ${session.blocks.map { it.speaker }.filter { it>=0 }.distinct().size} говорящих"
            } catch(e:Throwable) {
                JobState.status=if(JobState.cancel)"Обработка отменена. Распознанная часть сохранена." else "Ошибка: ${e.message?:e.javaClass.simpleName}"
                File(filesDir,"diagnostics.txt").writeText("VoiceSkip Fold7 GigaAM 1.0\nAndroid ${Build.VERSION.RELEASE} / ${Build.MODEL}\n"+e.stackTraceToString())
            } finally {
                JobState.recording=false;JobState.busy=false
                if(wake?.isHeld==true)wake?.release();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf()
            }
        }.start()
        return START_NOT_STICKY
    }
    override fun onTimeout(startId:Int,fgsType:Int) { JobState.cancel=true;stopForeground(STOP_FOREGROUND_REMOVE);stopSelf() }
    private fun prepareModels():File {
        val dir=File(filesDir,"models-v1").also { it.mkdirs() };val list=JSONArray(assets.open("models/manifest.json").bufferedReader().use { it.readText() })
        for(i in 0 until list.length()) {
            check(!JobState.cancel){"Обработка отменена"}
            val row=list.getJSONObject(i);val file=File(dir,row.getString("name"))
            if(!file.exists() || file.length()!=row.getLong("size")) {
                val temp=File(dir,"${file.name}.tmp")
                assets.open("models/${file.name}").use { input -> temp.outputStream().use { input.copyTo(it) } }
                val digest=MessageDigest.getInstance("SHA-256")
                temp.inputStream().use { input -> val buf=ByteArray(1024*1024);while(true){val n=input.read(buf);if(n<0)break;digest.update(buf,0,n)} }
                val actual=digest.digest().joinToString(""){"%02x".format(it)}
                require(actual==row.getString("sha256")){"Повреждена модель ${file.name}"}
                check(temp.renameTo(file)){"Не удалось подготовить модель"}
            }
        };return dir
    }
    private fun record(file:File) {
        val size=maxOf(AudioRecord.getMinBufferSize(16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT),32000)
        val recorder=AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,size)
        require(recorder.state==AudioRecord.STATE_INITIALIZED){"Не удалось открыть микрофон"}
        try {
            recorder.startRecording();val buffer=ShortArray(8000);var total=0L
            DataOutputStream(BufferedOutputStream(file.outputStream())).use { out ->
                while(!JobState.stopRecord && !JobState.cancel) {
                    val n=recorder.read(buffer,0,buffer.size);check(n>0){"Микрофон перестал передавать звук"}
                    for(i in 0 until n)out.writeFloat(buffer[i]/32768f)
                    total+=n;JobState.status="Запись · ${total/16000/60} мин ${total/16000%60} с"
                    if(total>=2L*60*60*16000)JobState.stopRecord=true
                }
            }
        } finally {runCatching{recorder.stop()};recorder.release()}
    }
    internal fun process(samples:FloatArray,dir:File,session:Session) {
        fun path(n:String)=File(dir,n).absolutePath
        val recognizer=OfflineRecognizer(config=OfflineRecognizerConfig(modelConfig=OfflineModelConfig(
            transducer=OfflineTransducerModelConfig(encoder=path("gigaam_v3_e2e_rnnt_encoder_int8.onnx"),decoder=path("gigaam_v3_e2e_rnnt_decoder.onnx"),joiner=path("gigaam_v3_e2e_rnnt_joint.onnx")),
            tokens=path("gigaam_v3_e2e_rnnt_tokens.txt"),modelType="nemo_transducer",numThreads=4,provider="cpu")))
        val words=mutableListOf<Word>();val raw=StringBuilder();var pos=0
        try {
            while(pos<samples.size) {
                check(!JobState.cancel){"Обработка отменена"}
                val end=Transcript.chunkEnd(samples,pos)
                JobState.status="Распознавание · ${pos*100L/samples.size}%"
                val stream=recognizer.createStream()
                try {
                    val chunk=FloatArray(end-pos+16000);samples.copyInto(chunk,0,pos,end)
                    stream.acceptWaveform(chunk,16000);recognizer.decode(stream);val r=recognizer.getResult(stream)
                    if(r.text.isNotBlank()) {
                        raw.append("[${pos/16000.0}–${end/16000.0}] ${r.text}\n\n")
                        words.addAll(Transcript.words(r.tokens,r.timestamps,pos/16000f,end/16000f))
                    }
                } finally {stream.release()}
                session.raw=raw.toString();session.blocks.clear();session.blocks.addAll(Transcript.blocks(words,emptyList()));session.save(this)
                pos=end
            }
        } finally {recognizer.release()}
        require(words.isNotEmpty()){ "Речь не распознана. Проверьте громкость и содержимое записи." }
        check(!JobState.cancel){"Обработка отменена"}
        JobState.status="Разделение говорящих"
        val diarization=OfflineSpeakerDiarization(config=OfflineSpeakerDiarizationConfig(
            segmentation=OfflineSpeakerSegmentationModelConfig(pyannote=OfflineSpeakerSegmentationPyannoteModelConfig(model=path("segmentation.onnx")),numThreads=4),
            embedding=SpeakerEmbeddingExtractorConfig(model=path("embedding.onnx"),numThreads=4),
            clustering=FastClusteringConfig(threshold=.9f),minDurationOn=.2f,minDurationOff=.5f))
        val turns=try {diarization.processWithCallback(samples,{done,total,_ -> JobState.status="Разделение говорящих · ${done*100/maxOf(1,total)}%";if(JobState.cancel)1 else 0}).map { Turn(it.start,it.end,it.speaker) }} finally {diarization.release()}
        check(!JobState.cancel){"Обработка отменена"}
        // Stable labels follow the order of first speech, never inferred legal roles.
        val ids=turns.sortedBy{it.start}.map{it.speaker}.distinct().withIndex().associate{it.value to it.index}
        session.blocks.clear();session.blocks.addAll(Transcript.blocks(words,turns.map{it.copy(speaker=ids.getValue(it.speaker))}))
        session.complete=true;session.save(this)
    }
}
''',

'app/src/test/java/com/voiceskip/fold7/court/TranscriptTest.kt': r'''package com.voiceskip.fold7.court
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.zip.ZipInputStream
import javax.xml.parsers.DocumentBuilderFactory

class TranscriptTest {
 @Test fun overlapAndGapsAreUnknown() {
  val w = Word(1f, 2f, "слово")
  assertEquals(-1, Transcript.speaker(w, emptyList()))
  assertEquals(-1, Transcript.speaker(w, listOf(Turn(0f, 3f, 0), Turn(1.5f, 3f, 1))))
  assertEquals(0, Transcript.speaker(w, listOf(Turn(0f, 3f, 0))))
 }
 @Test fun subwordsRetainLegalTokens() {
  val w = Transcript.words(arrayOf(" Не", " при", "з", "на", "ю", " 25", "."), floatArrayOf(0f,.1f,.2f,.3f,.4f,.6f,.7f),10f,11f)
  assertEquals(listOf("Не", "признаю", "25."), w.map { it.text })
  assertEquals(10f,w[0].start)
 }
 @Test fun windowsCoverWithoutGapsAndStayBelow25Seconds() {
  for (seconds in listOf(1, 24, 25, 26, 27, 49, 51, 426)) {
   val a=FloatArray(seconds*16000); var pos=0
   while(pos<a.size) { val next=Transcript.chunkEnd(a,pos); assertTrue(next>pos);assertTrue(next-pos<=400000); pos=next }
   assertEquals(a.size,pos)
  }
 }
 @Test fun docxIsValidAndEscapesNamesAndLegalText() {
  val out=ByteArrayOutputStream(); val text="Не признаю иск: 1 250,50 руб. <ст. 12> & № 3"
  Transcript.docx(out,listOf(Block(0f,0,text))) { "Иванов & Петров" }
  val z=ZipInputStream(out.toByteArray().inputStream()); var count=0
  while(true) { val e=z.nextEntry?:break; val bytes=z.readBytes(); DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(bytes.inputStream()); if(e.name=="word/document.xml") assertTrue(String(bytes).contains("1 250,50 руб.")); count++ }
  assertEquals(3,count)
 }
}
''',

'build.gradle.kts': r'''plugins {
    id("com.android.application") version "8.11.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
}
''',

'gradle.properties': r'''org.gradle.jvmargs=-Xmx4g -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
''',

'settings.gradle.kts': r'''pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "VoiceSkipCourt"
include(":app")
''',

}

ROOT = Path('court')
for name, text in SOURCES.items():
    p = ROOT / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding='utf8')
DOWNLOADS = [{'name': 'gigaam_v3_e2e_rnnt_encoder_int8.onnx', 'sha256': '2cac62d0c270bd128f898f2be1a2d34780d524a6e9483888ebac7b00f97410f1', 'size': 318995997, 'url': 'https://huggingface.co/pantinor/gigaam-v3/resolve/9890aad209f38eb48b2c853e42518dbbfcc5da0a/gigaam_v3_e2e_rnnt_encoder_int8.onnx'}, {'name': 'gigaam_v3_e2e_rnnt_decoder.onnx', 'sha256': '781971998e6a355d6a714f6932a30eab295e7ba0d14fd7e0f78c83b87e811860', 'size': 4600058, 'url': 'https://huggingface.co/pantinor/gigaam-v3/resolve/9890aad209f38eb48b2c853e42518dbbfcc5da0a/gigaam_v3_e2e_rnnt_decoder.onnx'}, {'name': 'gigaam_v3_e2e_rnnt_joint.onnx', 'sha256': '602ff7017a93311aad34df1437c8d7f49911353c13d6eae7a6ee7b041339465c', 'size': 2712896, 'url': 'https://huggingface.co/pantinor/gigaam-v3/resolve/9890aad209f38eb48b2c853e42518dbbfcc5da0a/gigaam_v3_e2e_rnnt_joint.onnx'}, {'name': 'gigaam_v3_e2e_rnnt_tokens.txt', 'sha256': '7ddf22514c42c531358182c81446a8159771e9921019f09ae743ea622d40221d', 'size': 13353, 'url': 'https://huggingface.co/pantinor/gigaam-v3/resolve/9890aad209f38eb48b2c853e42518dbbfcc5da0a/gigaam_v3_e2e_rnnt_tokens.txt'}, {'name': 'sherpa.aar', 'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-1.13.8.aar', 'sha256': '633c24321e06b1fe79feafa03ea16cbc0f8a286641e2da3559bac91bdb13bd96', 'size': 50129134}, {'name': 'segmentation.tar.bz2', 'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2', 'sha256': '24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488', 'size': 6958444}, {'name': 'embedding.onnx', 'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx', 'sha256': '1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b', 'size': 39593761}, {'name': 'example.wav', 'url': 'https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav', 'sha256': 'd8aaaa18a5098d7c6de0595ae7ac1e64cacd0d4022af3595213bdaf23be77e69'}]

def download(row):
    name = row['name']
    if name == 'sherpa.aar': dest = ROOT / 'app/libs' / name
    elif name == 'example.wav': dest = ROOT / 'app/src/androidTest/assets' / name
    else: dest = ROOT / 'app/src/main/assets/models' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print('Download', name, flush=True)
        urllib.request.urlretrieve(row['url'], dest)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == row['sha256'], name
    return dest
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    list(pool.map(download, DOWNLOADS))
models = ROOT / 'app/src/main/assets/models'
licenses = ROOT / 'app/src/main/assets/licenses'; licenses.mkdir(exist_ok=True)
with tarfile.open(models / 'segmentation.tar.bz2') as archive:
    member = next(m for m in archive.getmembers() if m.name.endswith('/model.onnx'))
    data = archive.extractfile(member).read()
    assert hashlib.sha256(data).hexdigest() == '220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079'
    (models / 'segmentation.onnx').write_bytes(data)
    license_member = next(m for m in archive.getmembers() if m.name.endswith('/LICENSE'))
    (licenses / 'pyannote.txt').write_bytes(archive.extractfile(license_member).read())
(models / 'segmentation.tar.bz2').unlink()
manifest = [dict(name=p.name, size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(models.iterdir()) if p.is_file() and p.name != 'manifest.json']
(models / 'manifest.json').write_text(json.dumps(manifest, indent=2))
licenses = ROOT / 'app/src/main/assets/licenses'; licenses.mkdir(exist_ok=True)
license_urls = {
    'GigaAM.txt': 'https://raw.githubusercontent.com/salute-developers/GigaAM/main/LICENSE',
    'sherpa-onnx.txt': 'https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.8/LICENSE',
    'onnxruntime.txt': 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE',
    '3D-Speaker.txt': 'https://raw.githubusercontent.com/modelscope/3D-Speaker/main/LICENSE',
}
for name, url in license_urls.items():
    urllib.request.urlretrieve(url, licenses / name)
(ROOT / 'provenance.json').write_text(json.dumps(DOWNLOADS, indent=2))
print('Prepared source and verified all model checksums', flush=True)
