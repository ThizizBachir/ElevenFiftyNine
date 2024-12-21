import 'dart:io' ;
import 'dart:typed_data';

import 'package:flutter_3d_controller/flutter_3d_controller.dart';
import 'package:flutter/material.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:screenshot/screenshot.dart';
import 'package:share_plus/share_plus.dart';

class ThreeDRenderdingScreen extends StatefulWidget {
  static const routeName = '3d_screen';

  const ThreeDRenderdingScreen({
    Key? key,
  }) : super(key: key);

  @override
  State<ThreeDRenderdingScreen> createState() => _ThreeDRenderdingScreenState();
}

class _ThreeDRenderdingScreenState extends State<ThreeDRenderdingScreen> {
  late String asset;
  Flutter3DController objectController = Flutter3DController();
  ScreenshotController screenshotController = ScreenshotController();

  bool isLoading = true;
  
Future<void> saveImage(Uint8List bytes) async{
  final name = 'screenshot_${DateTime.now()}';
  await Permission.storage.request();
  await ImageGallerySaver.saveImage(bytes,name: name);

}

Future<void> saveAndShare( Uint8List bytes)async{
  final directory = await getExternalStorageDirectory();
  final image = File('${directory!.path}/${DateTime.now()}.png');
  image.writeAsBytes(bytes);
  await Share.shareXFiles([XFile(image.path)]);
}


Widget flutter3D() {
  return Flutter3DViewer(
                      onProgress: (progressValue) {
                        if (progressValue == 1.0) {
                          setState(() {
                            isLoading = false;
                          });
                        }
                        return const SizedBox.shrink();
                      },
                      onLoad: (modelAddress) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      onError: (error) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      src: asset,
                      activeGestureInterceptor: true,
                      progressBarColor: Colors.orange,
                      enableTouch: true,
                      controller: objectController,
                    );
}
  @override
  Widget build(BuildContext context) {
    var asset = 'assets/mehdi_chekir.glb'; 
    return Screenshot(
      controller: screenshotController,
      child: Scaffold(
        appBar: AppBar(title: const Text('3D Model Viewer')),
        body: Stack(
          children: [
            Column(
              children: [
                if (asset.contains('.obj'))
                  Expanded(
                    child: Flutter3DViewer.obj(
                      scale: 1.0, 
                      cameraX: 0,
                      cameraY: 0,
                      cameraZ: 10,
                      onProgress: (progressValue) {
                        if (progressValue == 1.0) {
                          setState(() {
                            isLoading = false;
                          });
                        }
                        return const SizedBox.shrink();
                      },
                      onLoad: (modelAddress) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      onError: (error) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      src: asset,
                    ),
                  ),
                if (asset.contains('.glb') || asset.contains('.gltf'))
                  SizedBox(
                    height: 500,
                    child: 
                    Flutter3DViewer(
                      onProgress: (progressValue) {
                        if (progressValue == 1.0) {
                          setState(() {
                            isLoading = false;
                          });
                        }
                        return const SizedBox.shrink();
                      },
                      onLoad: (modelAddress) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      onError: (error) {
                        setState(() {
                          isLoading = false;
                        });
                      },
                      src: asset,
                      activeGestureInterceptor: true,
                      progressBarColor: Colors.orange,
                      enableTouch: true,
                      controller: objectController,
                    ),
                  ),
                  const SizedBox(height: 50,),
                  Column(
                    children: [
                      TextButton(
                    style: TextButton.styleFrom(
                  backgroundColor: Colors.deepOrange,
                  foregroundColor: Colors.white,
                  minimumSize: const Size(240, 60),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8.0), 
            ),
                ),
                    onPressed:()async {
                        await screenshotController.capture().then((bytes){
                          if(bytes!= null){
                            saveImage(bytes);
                            saveAndShare(bytes);
                          }
                        });
                     } ,
                    child:const  Text('ScreenShot and Share',style: TextStyle(fontSize: 20, color: Colors.white),
                    )),
                    const SizedBox(height: 20,),
                    TextButton(
                    style: TextButton.styleFrom(
                  backgroundColor: Colors.deepOrange,
                  foregroundColor: Colors.white,
                  minimumSize: const Size(240, 60),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8.0), 
            ),
                ),
                    onPressed:(){
                        objectController.playAnimation(animationName: 'pointing forward');
                      },
                    child:const  Text('Animate Model',style: TextStyle(fontSize: 20, color: Colors.white),
                    )),
                    ],
                  )
              ],
            ),
            if (isLoading)
              const Center(
                child: CircularProgressIndicator(
                  color: Colors.deepOrange,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
