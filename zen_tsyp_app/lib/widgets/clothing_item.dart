import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/screens/item_detials_screen.dart';

class ClothingItem extends StatelessWidget {
  final String imageAsset;
  final double price;
  final String size;
  final String title;
  final String description;
  const ClothingItem({required this.title, required this.imageAsset,required this.price,required this.size,required this.description, super.key});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
          Navigator.of(context).pushNamed(ItemDetialsScreen.routeName,arguments: {
            'asset':imageAsset.toString(),
            'title':title.toString(),
            'size':size.toString(),
            'price':price.toString(),
            'desrciption':description.toString()
          });
        },
      child: ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: GridTile(
          footer: GridTileBar(title: Text(title,textAlign: TextAlign.center,),
          backgroundColor: Colors.black87,
          leading: Text('$price DT',style: const TextStyle(color: Colors.deepOrange),),
          trailing: Text(size,style: const TextStyle(color: Colors.deepOrange),),
          ),
          
          child: 
               Image.asset(imageAsset),
          
         
        ),
      ),
    );
  }
}