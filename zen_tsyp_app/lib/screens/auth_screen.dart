
import 'package:zen_tsyp_app/widgets/auth_form.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/clothing_screnn.dart';

class AuthScreen extends StatefulWidget {
  static const routeName = '/authScreen';
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  var isLoading = false;
  final auth = FirebaseAuth.instance;

  void submitAuthForm(
    String email,
    String password,
    String userName,
    bool isLogin,
    BuildContext ctx,
  ) async {
    UserCredential? userCredential;
    try {
      setState(() {
        isLoading = true;
      });
      
      if (isLogin) {
        userCredential = await auth.signInWithEmailAndPassword(
          email: email,
          password: password,
        );
      } else {
        userCredential = await auth.createUserWithEmailAndPassword(
          email: email,
          password: password,
        );
        if (userCredential.user != null) {
          await FirebaseFirestore.instance
              .collection('users')
              .doc(userCredential.user!.uid)
              .set({
            'username': userName,
            'email': email,
          });
        }
      }
      if (userCredential.user != null) {
         if (!mounted) return;
       Navigator.of(ctx).pushReplacementNamed(ClothingScreen.routeName);
}   

    } on FirebaseAuthException catch (err) {
      setState(() {
        isLoading=false;
      });
      String message = 'An error occurred, Please check your credentials.';
      if (err.message != null) {
        message = err.message!;
      }
      
      if (!mounted) return;
      ScaffoldMessenger.of(ctx).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
        ),
      );
      
    } catch (err) {
      if (!mounted) return;
      ScaffoldMessenger.of(ctx).showSnackBar(
        const SnackBar(
          content: Text('An error occurred. Please try again later.'),
          backgroundColor: Colors.red,
        ),
      );
      
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: AuthForm(submitAuthForm, isLoading)
    );
  }
}