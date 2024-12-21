import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/3d_rendring_screen.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/clothing_screnn.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/man_cothing_screen.dart';
import 'package:zen_tsyp_app/screens/face_body_validation_screen.dart';
import 'package:zen_tsyp_app/screens/item_detials_screen.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';


late FirebaseFirestore firestore;
late FirebaseAuth auth;
late FirebaseStorage storage;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
      options:const  FirebaseOptions(
        apiKey: 'AIzaSyAIafS8lcMaUwgDmFncdmQqPHXp6BRYIwE',
    appId: '1:22686388340:android:faa6b0f5916bf95c170121',
    messagingSenderId: '22686388340',
    projectId: 'zen-challenge',
    storageBucket: 'zen-challenge.firebasestorage.app',
      ),
    );
  

  runApp(MyApp());
}


class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home:  const ClothingScreen(),
      routes: {
        ItemDetialsScreen.routeName:(context)=>const ItemDetialsScreen(),
        ManCothingScreen.routeName:(context)=>const ManCothingScreen(),
        FaceBodyValidationScreen.routeName:(context)=>const FaceBodyValidationScreen(),
        ThreeDRenderdingScreen.routeName:(context)=>const ThreeDRenderdingScreen(),
        ClothingScreen.routeName:(context)=>const ClothingScreen()
      },
    );
  }
}

