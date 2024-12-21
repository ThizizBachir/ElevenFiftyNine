import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:zen_tsyp_app/screens/3d_rendring_screen.dart';
class FaceBodyValidationScreen extends StatefulWidget {
  static const routeName= '/face_body_validation_screen';

  const FaceBodyValidationScreen({super.key});
  @override
  _FaceBodyValidationScreenState createState() =>
      _FaceBodyValidationScreenState();
}

class _FaceBodyValidationScreenState extends State<FaceBodyValidationScreen> {
  late CameraController _controller;
  late List<CameraDescription> cameras;
  bool _isInitialized = false;
  String _step = 'face'; 
  String _message = 'Align your face within the circle';

  final double boxTop = 10; 
  final double boxLeft = 10; 
  final double boxWidth = 10000; 
  final double boxHeight = 10000; 

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    cameras = await availableCameras();

    CameraDescription frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () =>
          cameras.first, 
    );

  
    _controller = CameraController(frontCamera, ResolutionPreset.high);
    await _controller.initialize();

    
    _startFaceDetection();

    setState(() {
      _isInitialized = true;
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<File?> _startFaceDetection() async {
    while (_controller.value.isInitialized) {
      await Future.delayed(
          const Duration(seconds: 1)); 

      final image = await _controller.takePicture();
      final imageFile = File(image.path);

      bool isFaceValid = await _validateFace(imageFile);
      if (isFaceValid) {
        setState(() {
          _step = 'body';
          _message = 'Align your upper body within the rectangle';
        });

        _showFaceValidationDialog(imageFile, "Face Detection", "Face");
        return imageFile;
    
      } else {
        return File('"C:/Users/LENOVO/SUPCOM/formation flutter/zen_tsyp_app/assets/error.jfif"');
      }
    }
  }

  Future<void> _showFaceValidationDialog(
      File imageFile, String title, String desc) async {
    return showDialog<void>(
      context: context,
      barrierDismissible: false, 
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(title),
          content: SingleChildScrollView(
            child: ListBody(
              children: <Widget>[
                Text('Your $desc has been successfully detected.'),
                desc == "Face"
                    ? const Text('Proceeding to upper body alignment...')
                    : Container(),
              ],
            ),
          ),
          actions: <Widget>[
            TextButton(
              child: const Text('OK'),
              onPressed: () {
                desc == "Face"
                    ? {
                        Navigator.of(context).pop(), 
                        _startBodyDetection(imageFile)
                      }
                    : Navigator.of(context).pushReplacementNamed(ThreeDRenderdingScreen.routeName); 
              },
            ),
          ],
        );
      },
    );
  }

  Future<bool> _validateFace(File imageFile) async {
    final inputImage = InputImage.fromFile(imageFile);
    final faceDetector = GoogleMlKit.vision.faceDetector(FaceDetectorOptions(
      enableContours: true,
      enableClassification: true,
    ));
    final faces = await faceDetector.processImage(inputImage);

    faceDetector.close();
    return faces.isNotEmpty; 
  }

  Future<File?> _startBodyDetection(File imageFile) async {
    bool isBodyValid = await _validateBody(imageFile);
    if (isBodyValid) {
      _showFaceValidationDialog(imageFile, "Body Detection", "Body");
      return imageFile;
    } else {
      return File('"C:/Users/LENOVO/SUPCOM/formation flutter/zen_tsyp_app/assets/error.jfif"');
    }
  }

  Future<bool> _validateBody(File imageFile) async {
    final inputImage = InputImage.fromFile(imageFile);
    final poseDetector = GoogleMlKit.vision.poseDetector();
    final poses = await poseDetector.processImage(inputImage);

    bool isValid = false;

    for (Pose pose in poses) {
      bool upperBodyInsideBox = true;

      PoseLandmark? leftShoulder =
          pose.landmarks[PoseLandmarkType.leftShoulder];
      PoseLandmark? rightShoulder =
          pose.landmarks[PoseLandmarkType.rightShoulder];
      if (!_isLandmarkInsideBox(leftShoulder) ||
          !_isLandmarkInsideBox(rightShoulder)) {
        upperBodyInsideBox = false;
      }
      

      PoseLandmark? leftElbow = pose.landmarks[PoseLandmarkType.leftElbow];
      PoseLandmark? rightElbow = pose.landmarks[PoseLandmarkType.rightElbow];
      if (!_isLandmarkInsideBox(leftElbow) ||
          !_isLandmarkInsideBox(rightElbow)) {
        upperBodyInsideBox = false;
      }

      if (upperBodyInsideBox) {
        isValid = true;
     

        break; 
      }
    }

    poseDetector.close();
    return isValid;
  }

  bool _isLandmarkInsideBox(PoseLandmark? landmark) {
    if (landmark == null) return false;
    double x = landmark.x;
    double y = landmark.y;

    return x >= boxLeft &&
        x <= boxLeft + boxWidth &&
        y >= boxTop &&
        y <= boxTop + boxHeight;
  }


Future<String?> saveImagesToFolder(File image1, File image2) async {
  try {
    final appDirectory = await getApplicationDocumentsDirectory(); 
    final appPath= appDirectory.path;
    
    final directory = Directory('{$appPath/images}');
    if (!await directory.exists()) {
      await directory.create(recursive: true);
    }

    final image1Path = '${directory.path}/${image1.uri.pathSegments.last}';
    await image1.copy(image1Path);

   
    final image2Path = '${directory.path}/${image2.uri.pathSegments.last}';
    await image2.copy(image2Path);

    return directory.path;
  } catch (e) {
    return '';
  }
}


Future<File?> uploadImagesToAzure(String folderPath) async {
  try {
    final directory = Directory(folderPath);
    if (!directory.existsSync()) {
      throw Exception("The folder path does not exist.");
    }

    final imageFiles = directory
        .listSync()
        .whereType<File>()
        .where((file) => ['.png', '.jpg', '.jpeg']
            .contains(path.extension(file.path).toLowerCase()))
        .toList();

    if (imageFiles.length < 2) {
      throw Exception("The folder must contain at least two images.");
    }

    final uri = Uri.parse("<YOUR_AZURE_ENDPOINT>");
    final request = http.MultipartRequest("POST", uri);


    for (var i = 0; i < imageFiles.length; i++) {
      request.files.add(
        await http.MultipartFile.fromPath('file$i', imageFiles[i].path),
      );
    }

    final response = await request.send();

    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      final decodedResponse = json.decode(responseBody);

      final objFileUrl = decodedResponse['objFileUrl'];

      final objResponse = await http.get(Uri.parse(objFileUrl));
      if (objResponse.statusCode == 200) {
        final objFilePath = path.join(folderPath, 'result.obj');
        final objFile = File(objFilePath);
        await objFile.writeAsBytes(objResponse.bodyBytes);
        return objFile;
      } else {
        throw Exception("Failed to download the OBJ file.");
      }
    } else {
      throw Exception("Failed to upload images: ${response.statusCode}");
    }
  } catch (e) {
    debugPrint("Error uploading images to Azure: $e");
    return null;
  }
}



  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: CameraPreview(_controller),
          ),

          Positioned(
            top: 200, 
            left: 40,
            right: 40, 
            child: _step == 'face'
                ? Container(
                    width: 350, 
                    height: 350, 
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.transparent,
                      border: Border.all(color: Colors.red, width: 3),
                    ),
                  )
                : const SizedBox.shrink(),
          ),

          Positioned(
            top: 120, 
            left: 20,
            right: 20, 
            child: _step == 'body'
                ? Container(
                    width: boxWidth, 
                    height: 700,
                    decoration: BoxDecoration(
                      color: Colors.transparent,
                      border: Border.all(color: Colors.blue, width: 3),
                    ),
                  )
                : const SizedBox.shrink(),
          ),

          Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: Text(
              _message,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                backgroundColor: Colors.black54,
              ),
            ),
          ),
        ],
      ),
    );
  }

}
