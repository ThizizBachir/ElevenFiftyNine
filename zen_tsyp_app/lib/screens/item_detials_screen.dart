import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/man_cothing_screen.dart';
import 'package:zen_tsyp_app/screens/face_body_validation_screen.dart';

class ItemDetialsScreen extends StatelessWidget {
  static const routeName ='/item_detials_screen';

  const ItemDetialsScreen({super.key});
  @override
  Widget build(BuildContext context) {
    var passedData= ModalRoute.of(context)!.settings.arguments as Map<String,String>;
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () {
            Navigator.of(context).pushReplacementNamed(ManCothingScreen.routeName);
          },
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Stack(
                children: [
                  Image.asset(
                    passedData['asset']!, 
                    height: 200,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Text(
              passedData['title']!,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            const Row(
              children: [
                Icon(Icons.star, color: Colors.orange, size: 16),
                SizedBox(width: 4),
                Text(
                  '5.0 (124 reviews)',
                  style: TextStyle(color: Colors.grey, fontSize: 14),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const SizedBox(height: 16),
            const Text(
              'DESCRIPTION:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              passedData['desrciption']!,
              style: const TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 50),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
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
                    onPressed: (){
                        Navigator.of(context).pushReplacementNamed(FaceBodyValidationScreen.routeName);
                      
        
                      },
                    child:const  Text('Virtual Try On',style: TextStyle(fontSize: 20, color: Colors.white),
                    ))
              ],
            )
          ],
        ),
      ),
    );
  }
}
