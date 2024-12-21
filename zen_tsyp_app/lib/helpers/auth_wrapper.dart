import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/auth_screen.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/clothing_screnn.dart';

class AuthWrapper extends StatelessWidget {
  static const routeName = '/';
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return const ClothingScreen();
        }else{
        return const AuthScreen();
        }
      },
    );
  }
}