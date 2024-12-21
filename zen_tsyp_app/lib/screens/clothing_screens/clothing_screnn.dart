import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/man_cothing_screen.dart';
import 'package:zen_tsyp_app/screens/clothing_screens/woman_clothing_screen.dart';

class ClothingScreen extends StatelessWidget {
  static const routeName = '/clothing_screen';
  const ClothingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text(
            'Choose Your Clothes',
            style: TextStyle(fontSize: 22),
            textAlign: TextAlign.center,
          ),
          centerTitle: true,
          bottom: const PreferredSize(
            preferredSize: Size.fromHeight(50.0), 
            child: TabBar(
              labelColor: Colors.black,
              indicatorColor: Colors.deepOrange,
              tabs: [
                Tab(
                  icon: Icon(Icons.man, color: Colors.deepOrange),
                  text: 'Men',
                ),
                Tab(
                  icon: Icon(Icons.woman, color: Colors.deepOrange),
                  text: 'Women',
                ),
              ],
            ),
          ),
        ),
        body: const TabBarView(
          children: [
            ManCothingScreen(),
            WomanClothingScreen(),
          ],
        ),
      ),
    );
  }
}
