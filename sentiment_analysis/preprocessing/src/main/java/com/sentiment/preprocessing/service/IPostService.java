package com.sentiment.preprocessing.service;
import com.sentiment.preprocessing.model.Post;
import java.util.List;


public interface IPostService {
    List<Post> findAll();
}